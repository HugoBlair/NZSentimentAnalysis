import os

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
@st.cache_data(ttl=600)
def load_data():
    df_submission = query_bigquery(TABLE_SUBMISSION)
    df_comment = query_bigquery(TABLE_COMMENT)
    return df_submission, df_comment


# Display Streamlit App
def main():
    st.title("Reddit Sentiment Analysis Dashboard")
    st.write("This dashboard displays sentiment analysis of Reddit data fetched from Google BigQuery.")

    # Sidebar
    st.sidebar.header("Filters")

    # Date range filter
    start_date = st.sidebar.date_input("Start Date")
    end_date = st.sidebar.date_input("End Date")

    subreddits = ["wellington", "auckland", "thetron", "chch", "dunedin", "tauranga",
                  "newzealand", "palmy", "napier", "newplymouth", "hawkesbay", "NelsonNZ", "queenstown"]
    selected_subreddits = st.sidebar.multiselect("Select Subreddits", subreddits, default=subreddits)

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
    df_submission_filtered = df_submission[
        df_submission['subreddit'].isin(selected_subreddits) &
        df_submission['topic'].isin(selected_topics)
        ]

    # Beginning of Graphing

    # Create Plotly Figure for comparing topic distributions across cities
    fig = go.Figure()
    for subreddit in selected_subreddits:
        topic_counts = df_submission_filtered[df_submission_filtered['subreddit'] == subreddit]['topic'].value_counts(

        ).reset_index()
        topic_counts.columns = ['topic', 'count']
        fig.add_trace(go.Bar(
            x=topic_counts['topic'],
            y=topic_counts['count'],
            name=subreddit
        ))

    # Update figure layout for better readability
    fig.update_layout(
        title='Distribution of Topics of Submissions by Subreddit',
        xaxis_title='Topic',
        yaxis_title='Count',
        barmode='group'
    )

    # Display the plot
    st.plotly_chart(fig)

    # Polarity and subjectivity scatter plot
    st.subheader("Polarity and Subjectivity Scatter Plot")

    fig = px.scatter(df_submission, x='title_polarity', y='title_subjectivity', color='subreddit',
                     title='Polarity vs. Subjectivity of Submission Titles')
    st.plotly_chart(fig)

    st.write("Click this button if you wish to see the raw data")
    # Display raw data
    if st.checkbox("Show raw data"):
        st.subheader("Submissions Data")
        st.write(df_submission)
        st.subheader("Comments Data")
        st.write(df_comment)


if __name__ == "__main__":
    main()
