import json
import os
import traceback
from typing import Dict, Any

import praw
import re
import spacy
from google.cloud.bigquery import DatasetReference
from spacytextblob.spacytextblob import SpacyTextBlob
import requests
from google.cloud import bigquery
from datetime import datetime
from transformers import pipeline
import torch

classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

'''
Additional required files:
praw.ini
GCP credentials json
'''

'''
Creating BigQuery tables if they don't exist
I do realize that a non-normalized database schema is not entirely ideal, but I decided that for ease of use with a small project like this,
it is easier to have my database more compact
'''


def create_tables():
    print("Creating BigQuery tables")
    schemas = {
        'submission': [
            bigquery.SchemaField("submission_id", "STRING", mode="REQUIRED"),
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
            bigquery.SchemaField("topic", "STRING"),
        ],
        'comment': [
            bigquery.SchemaField("comment_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("submission_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("polarity", "FLOAT"),
            bigquery.SchemaField("subjectivity", "FLOAT"),
            bigquery.SchemaField("date", "TIMESTAMP"),
            bigquery.SchemaField("entities", "RECORD", mode="REPEATED", fields=[
                bigquery.SchemaField("text", "STRING"),
                bigquery.SchemaField("label", "STRING"),

            ]),
            bigquery.SchemaField("topic", "STRING")
        ],
    }

    for table_name, schema in schemas.items():
        table_ref = dataset_ref.table(table_name)
        table = bigquery.Table(table_ref, schema=schema)
        client.create_table(table, exists_ok=True)


class SubmissionCache:
    def __init__(self, filename: str, bigquery_client: bigquery.Client, dataset_ref: bigquery.DatasetReference):
        self.filename = filename
        self.bigquery_client = bigquery_client
        self.dataset_ref = dataset_ref
        self.cache: Dict[str, Any] = self.load()

    def load(self) -> Dict[str, Any]:
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                return json.load(f)
        else:
            print(f"Cache file {self.filename} not found. Rebuilding from BigQuery...")
            return self.rebuild_from_bigquery()

    def rebuild_from_bigquery(self) -> Dict[str, Any]:
        cache = {"submissions": {}, "comments": {}}

        # Fetch submissions
        query = f"""
        SELECT submission_id, topic
        FROM `{self.bigquery_client.project}.{self.dataset_ref.dataset_id}.submission`
        """
        query_job = self.bigquery_client.query(query)
        for row in query_job:
            cache["submissions"][row.submission_id] = {"classification": row.topic}

        # Fetch comments
        query = f"""
        SELECT comment_id
        FROM `{self.bigquery_client.project}.{self.dataset_ref.dataset_id}.comment`
        """
        query_job = self.bigquery_client.query(query)
        for row in query_job:
            cache["comments"][row.comment_id] = {}

        self.save(cache)
        return cache

    def save(self, cache=None):
        if cache is None:
            cache = self.cache
        with open(self.filename, 'w') as f:
            json.dump(cache, f, indent=2)

    def submission_exists(self, submission_id: str) -> bool:
        return submission_id in self.cache["submissions"]

    def get_submission_classification(self, submission_id: str) -> Dict[str, Any]:
        if submission_id in self.cache["submissions"]:
            return self.cache["submissions"][submission_id]

    def comment_exists(self, comment_id: str) -> bool:
        return comment_id in self.cache["comments"]

    def add_submission(self, submission_id: str, data: Any):
        self.cache["submissions"][submission_id] = data
        self.save()

    def add_comment(self, comment_id: str):
        self.cache["comments"][comment_id] = {}
        self.save()

    def __len__(self) -> int:
        return len(self.cache["submissions"]) + len(self.cache["comments"])


# Initializing GCP bigquery
file_path = "nzsentimentanalysis-3b5ec598d3b7.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = file_path
client = bigquery.Client()
dataset_ref = DatasetReference("nzsentimentanalysis", "SentimentDB")
create_tables()
print("Initialized BigQuery")

# Initializing Submission Cache
CACHE_FILE = "submission_cache.json"
submission_cache = SubmissionCache(CACHE_FILE, client, dataset_ref)

# Loading spaCy
nlp = spacy.load("en_core_web_lg")
nlp.add_pipe('spacytextblob')
print("Initialized spaCy")


class NLPData:
    def __init__(self, polarity, subjectivity, entities):
        self.polarity = polarity
        self.subjectivity = subjectivity
        self.entities = entities


# Accessing the reddit API details from my praw.ini file (which is in the same directory as the script)
# This information is kept private and not included in the repository
reddit = praw.Reddit("bot1")

# Choosing the subreddits to search for comments in
subreddits = reddit.subreddit("wellington+auckland+thetron+chch+dunedin+tauranga+newzealand")

# List of acceptable labels for NLP tagging
labels = {
    "EVENT", "FAC", "GPE", "LAW", "LOC", "NORP", "ORG", "PERSON", "PRODUCT", "WORK_OF_ART"
}
classification_topics = ["Politics", "Economics", "Sports", "Technology", "Environment", "Health", "Education",
                         "Culture",
                         "Social Issues", "Transport", "Tourism", "Agriculture", "Foreign Affairs", "Housing",
                         "Justice and Law",
                         "Indigenous Affairs", "Other/Unknown"]


# Function to retrieve all submission ids from bigquery
# @param data_type either 'comment' or 'submission'
def get_ids_from_bigquery(data_type):
    try:
        data_type_id = data_type + "_id"
        query = """
        SELECT DISTINCT {}
        FROM `{}.{}.{}`
        """.format(data_type_id, client.project, dataset_ref.dataset_id, data_type)

        query_job = client.query(query)
        results = query_job.result()

        return {row.data_type_id for row in results}

    except Exception as e:
        print("Job Failed. Returning None")
        return []


# Function to save comments replied to
def exit_handler():
    print("Exiting")
    submission_cache.save()


def perform_nlp(doc):
    processed_doc = nlp(doc)
    entities = []
    for ent in processed_doc.ents:
        if ent.label_ in labels:
            entities.append({'text': ent.text, 'label': ent.label_})

    return NLPData(processed_doc._.blob.polarity, processed_doc._.blob.subjectivity, entities)


def perform_classification(doc):
    hypothesis_template = 'This text is about {}.'
    prediction = classifier(doc, classification_topics, hypothesis_template=hypothesis_template, multi_label=True)
    if prediction and prediction['labels']:
        return prediction['labels'][0]


def insert_submission(submission, title_nlp, body_nlp, topic):
    rows_to_insert = [{
        'submission_id': submission.id,
        'title_polarity': title_nlp.polarity,
        'title_subjectivity': title_nlp.subjectivity,
        'body_polarity': body_nlp.polarity,
        'body_subjectivity': body_nlp.subjectivity,
        'date': submission.created_utc,
        'subreddit': submission.subreddit.display_name,
        'title_entities': title_nlp.entities,
        'body_entities': body_nlp.entities,
        'topic': topic
    }]
    errors = client.insert_rows_json(dataset_ref.table('submission'), rows_to_insert)
    if errors:
        print(f"Errors inserting submission: {errors}")


def insert_comment(comment, comment_nlp, topic):
    rows_to_insert = [{
        'comment_id': comment.id,
        'submission_id': comment.submission.id,
        'polarity': comment_nlp.polarity,
        'subjectivity': comment_nlp.subjectivity,
        'date': comment.created_utc,
        'entities': comment_nlp.entities,
        'topic': topic
    }]
    errors = client.insert_rows_json(dataset_ref.table('comment'), rows_to_insert)
    if errors:
        print(f"Errors inserting comment: {errors}")


'''
# Removing stopwords from the text in order to make NLP quicker. This feature has been removed in 
order to maintain the accuracy of the subjectivity attribute.
def preprocess(text):
    output_text = ""
    for token in simple_preprocess(text):
        if token not in STOPWORDS:
            output_text += token + " "
    return output_text
'''

# Main loop to search for comments
try:
    print("Starting Sentiment Analysis Bot")

    for submission in subreddits.hot(limit=5):

        if not submission_cache.submission_exists(submission.id):
            submission_title_processed = perform_nlp(submission.title)
            submission_body_processed = perform_nlp(submission.selftext)
            submission_text = "Post Title:" + submission.title + "Body:" + submission.selftext + "Comments:"

            for comment in submission.comments:
                submission_text += comment.body + ". Comment:"
            classification_prediction = perform_classification(submission_text)
            insert_submission(submission, submission_title_processed, submission_body_processed,
                              classification_prediction)

            # Update cache
            submission_cache.add_submission(submission.id, classification_prediction)

        else:
            classification_prediction = submission_cache.get_submission_classification(submission.id)

        for comment in submission.comments:
            # print("Found new comment")
            if not submission_cache.comment_exists(comment.id):
                comment_processed = perform_nlp(comment.body)
                submission_cache.add_comment(comment.id)
                insert_comment(comment, comment_processed, classification_prediction)

    exit_handler()
    print("Finished Sentiment Analysis Successfully")


# Catching case where spaCy's model in not installed
except OSError:
    from spacy.cli import download

    download("en_core_web_lg")

# Exiting cleanly when the program is interrupted by the user
except KeyboardInterrupt:
    print("Stopped by user")
    exit_handler()

except Exception as e:
    print("Unknown Program Failure")
    print(traceback.print_exc())
    exit_handler()
