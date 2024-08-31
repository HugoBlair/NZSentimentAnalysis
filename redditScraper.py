import os

import praw
import re
import spacy
from google.cloud.bigquery import DatasetReference
from spacytextblob.spacytextblob import SpacyTextBlob
import requests
from google.cloud import bigquery
from datetime import datetime

'''
Additional required files:
praw.ini
GCP credentials json
'''

# Initalizing GCP bigquery

file_path = "nzsentimentanalysis-3b5ec598d3b7.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = file_path
client = bigquery.Client()
dataset_ref = DatasetReference("nzsentimentanalysis", "SentimentDB")

# Accessing the reddit API details from my praw.ini file (which is in the same directory as the script)
# This information is kept private and not included in the repository
reddit = praw.Reddit("bot1")

# Choosing the subreddits to search for comments in
subreddits = reddit.subreddit("wellington+auckland+thetron+chch+dunedin+tauranga+newzealand")

# List of acceptable labels for NLP tagging
labels = {
    'EVENT', 'FAC', 'GPE', 'LAW', 'LOC', 'NORP', 'ORG', 'PERSON', 'PRODUCT', 'WORK_OF_ART'
}

'''
Function to retrieve all submission ids from bigquery
@param data_type either 'comment' or 'submission'
'''


def get_ids_from_bigquery(data_type):
    data_type_id = data_type = "_id"
    query = """
    SELECT DISTINCT {}
    FROM `{}.{}.{}`
    """.format(data_type_id, client.project, dataset_ref.dataset_id, data_type)

    query_job = client.query(query)
    results = query_job.result()

    return {row.data_type_id for row in results}


# Load comments that have already been processed from file
def load_processed_comments():
    if not os.path.isfile("comments_processed.txt"):
        print("comments_processed.txt not found. Creating new file from bigquery data")
        comments_processed = get_ids_from_bigquery("comment")

    else:
        with open("comments_processed.txt", "r") as f:
            print("Loading comments_processed.txt")
            comments_processed = f.read()
            comments_processed = comments_processed.split("\n")
            comments_processed = list(filter(None, comments_processed))

    return comments_processed


# loading submissions that have already been processed from file
def load_processed_submissions():
    if not os.path.isfile("submissions_processed.txt"):
        print("Creating submissions_processed.txt")
        submissions_processed = get_ids_from_bigquery()
    else:
        with open("submissions_processed.txt", "r") as f:
            print("Loading submissions_processed.txt")
            submissions_processed = f.read()
            submissions_processed = submissions_processed.split("\n")
            submissions_processed = list(filter(None, submissions_processed))
    return submissions_processed


# Initializing comments_processed.txt and submissions_processed.txt
comments_processed = load_processed_comments()
print("Loaded processed_comments()")
submissions_processed = load_processed_submissions()
print("Loaded processed_submissions()")


# Function to save comments replied to
def exit_handler():
    print("Exiting")
    with open("comments_processed.txt", "w") as f:
        for comment_id in comments_processed:
            f.write(comment_id + "\n")
        print("Saved comments replied to")
    with open("submissions_processed.txt", "w") as f:
        for submission_id in submissions_processed:
            f.write(submission_id + "\n")
        print("Saved submissions replied to")


def perform_nlp(doc):
    entities = []
    for ent in doc.ents:
        print(ent.text, ent.label_)
        if ent.label_ in labels:
            entities.append({'Text': ent.text, 'Label': ent.label_})

    return NLPData(doc._.blob.polarity, doc._.blob.subjectivity, entities)


class NLPData:
    def __init__(self, polarity, subjectivity, entities):
        self.polarity = polarity
        self.subjectivity = subjectivity
        self.entities = entities


def insert_submission(submission, title_nlp, body_nlp):
    rows_to_insert = [{
        'submission_id': submission.id,
        'title': submission.title,
        'body': submission.selftext,
        'title_polarity': title_nlp.polarity,
        'title_subjectivity': title_nlp.subjectivity,
        'body_polarity': body_nlp.polarity,
        'body_subjectivity': body_nlp.subjectivity,
        'date': submission.created_utc,
        'subreddit': submission.subreddit.display_name,
        'title_entities': [{'text': ent[0], 'label': ent[1]} for ent in title_nlp.entities],
        'body_entities': [{'text': ent[0], 'label': ent[1]} for ent in body_nlp.entities],
    }]
    errors = client.insert_rows_json(dataset_ref.table('submission'), rows_to_insert)
    if errors:
        print(f"Errors inserting submission: {errors}")


def insert_comment(comment, comment_nlp):
    rows_to_insert = [{
        'comment_id': comment.id,
        'submission_id': comment.submission.id,
        'text': comment.body,
        'polarity': comment_nlp.polarity,
        'subjectivity': comment_nlp.subjectivity,
        'date': comment.created_utc,
        'entities': [{'text': ent[0], 'label': ent[1]} for ent in comment_nlp.entities],
    }]
    errors = client.insert_rows_json(dataset_ref.table('comment'), rows_to_insert)
    if errors:
        print(f"Errors inserting comment: {errors}")


'''
Creating BigQuery tables if they don't exist
I do realize that a non-normalized database schema is not entirely ideal, but I decided that for ease of use with a small project like this,
it is easier to have my database more compact
'''


def create_tables():
    schemas = {
        'submission': [
            bigquery.SchemaField("submission_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("title", "STRING"),
            bigquery.SchemaField("body", "STRING"),
            bigquery.SchemaField("title_polarity", "FLOAT"),
            bigquery.SchemaField("title_subjectivity", "FLOAT"),
            bigquery.SchemaField("body_polarity", "FLOAT"),
            bigquery.SchemaField("body_subjectivity", "FLOAT"),
            bigquery.SchemaField("date", "TIMESTAMP"),
            bigquery.SchemaField("subreddit", "STRING"),
            bigquery.SchemaField("title_entities", "RECORD", mode="REPEATED", fields=[
                bigquery.SchemaField("text", "STRING"),
                bigquery.SchemaField("label", "STRING"),
            ]),
            bigquery.SchemaField("body_entities", "RECORD", mode="REPEATED", fields=[
                bigquery.SchemaField("text", "STRING"),
                bigquery.SchemaField("label", "STRING"),
            ]),
        ],
        'comment': [
            bigquery.SchemaField("comment_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("submission_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("text", "STRING"),
            bigquery.SchemaField("polarity", "FLOAT"),
            bigquery.SchemaField("subjectivity", "FLOAT"),
            bigquery.SchemaField("date", "TIMESTAMP"),
            bigquery.SchemaField("entities", "RECORD", mode="REPEATED", fields=[
                bigquery.SchemaField("text", "STRING"),
                bigquery.SchemaField("label", "STRING"),
            ]),
        ],
    }

    for table_name, schema in schemas.items():
        table_ref = dataset_ref.table(table_name)
        table = bigquery.Table(table_ref, schema=schema)
        client.create_table(table, exists_ok=True)


# Main loop to search for comments
try:
    print("Starting Sentiment Analysis Bot")

    # Loading spaCy
    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe('spacytextblob')

    print("Initialized spaCy")

    create_tables()
    print("Initialized BigQuery")

    for submission in subreddits.hot(limit=3):
        if submission.id not in submissions_processed:
            submission_title_processed = perform_nlp(submission.title)
            submission_body_processed = perform_nlp(submission.selftext)
            submissions_processed.append(submission.id)
            insert_submission(submission, submission_title_processed, submission_body_processed)

        for comment in submission.comments:
            # print("Found new comment")
            if comment.id not in comments_processed:
                comment_processed = perform_nlp(comment)
                comments_processed.append(comment.id)
                insert_comment(comment, comment_processed)


# Catching case where spaCy's model in not installed
except OSError:
    from spacy.cli import download

    download("en_core_web_lg")
    # Exiting cleanly when the program is interrupted by the user
except KeyboardInterrupt:
    print("Stopped by user")
    exit_handler()
# Exiting cleanly when the program is interrupted by the user
except Exception as e:
    print("Unknown Program Failure")
    print(e)
    exit_handler()
