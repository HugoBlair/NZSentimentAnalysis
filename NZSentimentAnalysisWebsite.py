import os
from collections import Counter

import plotly.graph_objects
import streamlit as st
from google.cloud import bigquery
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Initialize BigQuery client
# Configure Google BigQuery client
file_path = "nzsentimentanalysis-3b5ec598d3b7.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = file_path
client = bigquery.Client()

# Define the dataset and table names
DATASET_ID = "SentimentDB"
PROJECT_ID = "nzsentimentanalysis"
TABLE_SUBMISSION = f"{PROJECT_ID}.{DATASET_ID}.submission"
TABLE_COMMENT = f"{PROJECT_ID}.{DATASET_ID}.comment"


# Query to get data from BigQuery
def query_bigquery(table_name):
    query = f"SELECT * FROM `{table_name}`"
    query_job = client.query(query)
    return query_job.to_dataframe()


# Load data from BigQuery
@st.cache_data(ttl=86400)
def load_data():
    df_submission = query_bigquery(TABLE_SUBMISSION)
    df_comment = query_bigquery(TABLE_COMMENT)
    return df_submission, df_comment


def plot_topic_trend(df_submission):
    df_submission['date'] = pd.to_datetime(df_submission['date'])
    df_submission['month'] = df_submission['date'].dt.to_period('M')
    topic_trend = df_submission.groupby(['month', 'topic']).size().unstack(fill_value=0)
    topic_trend = topic_trend.div(topic_trend.sum(axis=1), axis=0)

    fig = px.area(topic_trend.reset_index(), x='month', y=topic_trend.columns,
                  title='Topic Trends Over Time',
                  labels={'value': 'Proportion', 'variable': 'Topic'})
    st.plotly_chart(fig)


def plot_topic_heatmap(df_submission):
    # Calculate topic distribution across subreddits
    topic_distribution = df_submission.groupby(['subreddit', 'topic']).size().reset_index(name='percentage')

    # Create a pivot table for heatmap
    heatmap_data = topic_distribution.pivot_table(index='subreddit', columns='topic', values='percentage', fill_value=0)
    # Normalize each row to sum to 100 (percentage)
    heatmap_percentage = heatmap_data.div(heatmap_data.sum(axis=1), axis=0) * 100

    # Plot heatmap
    fig = px.imshow(heatmap_percentage,
                    aspect="auto",
                    color_continuous_scale='Viridis',
                    labels=dict(x="Topic", y="Subreddit", color="Percentage"),
                    title="Topic Distribution Heatmap for Subreddits")

    st.plotly_chart(fig)


def plot_topic_grouped_bar(df_submission, combined_counts, selected_topics):
    # Create a grouped bar chart to show the number of posts and comments per subreddit
    fig = px.bar(combined_counts, x='subreddit', y='count', color='type',
                 barmode='group', title='Number of Posts and Comments per Subreddit')

    # Create DataFrame for stacked bar chart with percentage distribution
    subreddit_topic_counts = df_submission.groupby(['subreddit', 'topic']).size().reset_index(name='count')
    total_counts_per_subreddit = subreddit_topic_counts.groupby('subreddit')['count'].transform('sum')
    subreddit_topic_counts['percentage'] = subreddit_topic_counts['count'] / total_counts_per_subreddit * 100

    # Create Plotly Figure for 100% stacked bar chart
    fig = go.Figure()
    for topic in selected_topics:
        topic_data = subreddit_topic_counts[subreddit_topic_counts['topic'] == topic]
        fig.add_trace(go.Bar(
            x=topic_data['subreddit'],
            y=topic_data['percentage'],
            name=topic
        ))

    # Update figure layout for 100% stacked bar
    fig.update_layout(
        title='Percentage Distribution of Topics in Submissions by Subreddit',
        xaxis_title='Subreddit',
        yaxis_title='Percentage',
        barmode='stack'
    )

    # Display the plot
    st.plotly_chart(fig)


def display_raw_data(df_submission, df_comment):
    st.write("Click this button if you wish to see the raw data")
    # Display raw data
    if st.checkbox("Show raw data"):
        st.subheader("Submissions Data")
        st.write(df_submission)
        st.subheader("Comments Data")
        st.write(df_comment)


# Sentiment vs. Subjectivity Across Topics
def plot_sentiment_subjectivity(df_submission):
    fig = px.scatter(df_submission, x='title_polarity', y='title_subjectivity',
                     color='topic', hover_data=['subreddit'],
                     title='Sentiment vs. Subjectivity Across Topics',
                     labels={'title_polarity': 'Sentiment Polarity',
                             'title_subjectivity': 'Subjectivity'})
    st.plotly_chart(fig)


# 1. Sentiment Distribution Across Subreddits
def plot_sentiment_distribution(df_submission):
    return None;


def normalize_entity(entity):
    # Ensure entity is a dictionary and has 'text' key
    if not isinstance(entity, dict) or 'text' not in entity:
        return None, None  # Skip this entity if it's not in the expected format

    # Convert to lowercase for case-insensitive comparison
    text = entity['text'].lower()

    # Combine specific entities
    if text in ['nz', 'new zealand']:
        return 'New Zealand', entity.get('label', 'GPE')
    elif text in ['maori', 'māori']:
        return 'Māori', entity.get('label', 'NORP')
    elif text in ['aus', 'australia']:
        return 'Australia', entity.get('label', 'GPE')
    else:
        return text.capitalize(), entity.get('label', 'UNKNOWN')  # Capitalize the first letter for display


def collect_entities(df_submission, df_comment):
    # Collect and normalize entities from submissions
    submission_entities = [normalize_entity(entity) for sublist in df_submission['title_entities'] for entity in sublist
                           if entity]
    submission_entities += [normalize_entity(entity) for sublist in df_submission['body_entities'] for entity in sublist
                            if entity]

    # Collect and normalize entities from comments
    comment_entities = [normalize_entity(entity) for sublist in df_comment['entities'] for entity in sublist if entity]

    # Combine all entities and remove None values
    return [entity for entity in (submission_entities + comment_entities) if entity[0] is not None]


def plot_entity_frequency(all_entities):
    # Count entity frequencies
    entity_freq = Counter(all_entities)

    # Create a dataframe for easier manipulation
    df_entities = pd.DataFrame(list(entity_freq.items()), columns=['entity_type', 'count'])
    df_entities[['entity', 'type']] = pd.DataFrame(df_entities['entity_type'].tolist(), index=df_entities.index)
    df_entities = df_entities.drop('entity_type', axis=1)

    # Sort by count in descending order
    df_entities = df_entities.sort_values('count', ascending=False).reset_index(drop=True)

    # Add a search bar
    search_term = st.text_input("Search for an entity:", "")

    # Filter entities based on search term
    if search_term:
        df_entities = df_entities[df_entities['entity'].str.contains(search_term, case=False)]

    # Take top 50 entities after filtering
    df_entities = df_entities.head(50)

    # Create a custom color map for entity types
    unique_types = df_entities['type'].unique()
    color_map = dict(zip(unique_types, px.colors.qualitative.Alphabet[:len(unique_types)]))

    # Create the color-coordinated bar chart
    fig = px.bar(df_entities, x='entity', y='count', color='type',
                 title='Top 50 Entities Mentioned in Submissions and Comments',
                 labels={'entity': 'Entity', 'count': 'Frequency', 'type': 'Entity Type'},
                 color_discrete_map=color_map)

    # Update layout for better readability
    fig.update_layout(
        xaxis_tickangle=-45,
        xaxis_title="",
        yaxis_title="Frequency",
        height=600,
        legend_title_text='Entity Type',
        xaxis={'categoryorder': 'total descending'}  # This ensures bars are sorted by height
    )

    st.plotly_chart(fig)


# Display Streamlit App
def main():
    st.title("Reddit Sentiment Analysis Dashboard")
    st.write("This dashboard displays sentiment analysis of Reddit data fetched from Google BigQuery.")

    # Sidebar
    st.sidebar.header("Filters")

    # Date range filter
    start_date = st.sidebar.date_input("Start Date")
    end_date = st.sidebar.date_input("End Date")
    min_submissions = st.sidebar.slider("Minimum number of submissions", min_value=0, max_value=1000, value=200,
                                        step=10)

    subreddits = ["Wellington", "auckland", "thetron", "chch", "dunedin", "Tauranga",
                  "newzealand", "palmy", "napier", "newplymouth", "hawkesbay", "NelsonNZ", "queenstown"]

    # List of acceptable labels for NLP tagging
    labels = {
        "EVENT", "FAC", "GPE", "LAW", "LOC", "NORP", "ORG", "PERSON", "PRODUCT", "WORK_OF_ART"
    }
    # List of topics for the LLM Classifier
    classification_topics = ["Politics", "Economics", "Sports", "Technology", "Environment", "Health", "Education",
                             "Culture",
                             "Social Issues", "Transport", "Tourism", "Agriculture", "Foreign Affairs", "Housing",
                             "Justice and Law",
                             "Indigenous Affairs", "Other/Unknown"]

    selected_topics = st.sidebar.multiselect("Select Topics", classification_topics, default=classification_topics)

    # Load data
    df_submission, df_comment = load_data()

    # Filter subreddits based on user's selection
    subreddit_counts = df_submission['subreddit'].value_counts()
    subreddits_filtered_by_slider = subreddit_counts[subreddit_counts >= min_submissions].index.tolist()

    # Filter out subreddits with fewer than 100 submissions
    subreddits_filtered = [sub for sub in subreddits if sub in subreddits_filtered_by_slider]
    selected_subreddits = st.sidebar.multiselect("Select Subreddits", subreddits, default=subreddits_filtered)

    df_submission_filtered = df_submission[
        df_submission['subreddit'].isin(selected_subreddits) &
        df_submission['topic'].isin(selected_topics)
        ]

    # Retrieving subreddit of each comment
    df_comment_enriched = df_comment.merge(df_submission[['submission_id', 'subreddit']], on='submission_id',
                                           how='left')
    df_comment_filtered = df_comment_enriched[df_comment_enriched['subreddit'].isin(selected_subreddits)]

    # Beginning of Graphing

    # Aggregate post and comment counts by subreddit
    submission_counts = df_submission_filtered['subreddit'].value_counts().reset_index()
    submission_counts.columns = ['subreddit', 'count']
    submission_counts['type'] = 'Submissions'

    comment_counts = df_comment_filtered['subreddit'].value_counts().reset_index()
    comment_counts.columns = ['subreddit', 'count']
    comment_counts['type'] = 'Comments'

    # Combine post and comment counts
    combined_counts = pd.concat([submission_counts, comment_counts])

    # plot_topic_trend(df_submission_filtered)
    plot_topic_grouped_bar(df_submission_filtered, combined_counts, selected_topics)
    plot_topic_heatmap(df_submission_filtered)

    plot_sentiment_subjectivity(df_submission_filtered)
    plot_sentiment_distribution(df_submission_filtered)

    all_entities = collect_entities(df_submission_filtered, df_comment_filtered)
    plot_entity_frequency(all_entities)

    display_raw_data(df_submission_filtered, df_comment_filtered)


if __name__ == "__main__":
    main()
