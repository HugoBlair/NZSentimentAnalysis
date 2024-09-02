import json
import os
import time
import traceback
from typing import Dict, Any
import praw
import re

import prawcore
import spacy
from google.cloud.bigquery import DatasetReference
from prawcore import ServerError
from spacytextblob.spacytextblob import SpacyTextBlob
import requests
from google.cloud import bigquery
from datetime import datetime, timezone
from transformers import pipeline
import torch

# Initialize the zero-shot classification pipeline
device = 0 if torch.cuda.is_available() else -1
torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
classifier = pipeline('zero-shot-classification', model='facebook/bart-base', device=device)

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
    """
    Create BigQuery tables for storing submission and comment data.

    This function defines the schema for 'submission' and 'comment' tables
    and creates them in BigQuery if they don't already exist.
    """
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
    """
    A class to manage caching of processed Reddit submissions and comments.

    This class handles loading, saving, and updating a local cache file
    to prevent reprocessing of already analyzed submissions and comments.
    """

    def __init__(self, filename: str, bigquery_client: bigquery.Client, dataset_ref: bigquery.DatasetReference):
        """
        Initialize the SubmissionCache.

        :param filename: Name of the cache file
        :param bigquery_client: BigQuery client instance
        :param dataset_ref: BigQuery dataset reference
        """
        self.filename = filename
        self.bigquery_client = bigquery_client
        self.dataset_ref = dataset_ref
        self.cache: Dict[str, Any] = self.load()

    def load(self) -> Dict[str, Any]:
        """
        Load the cache from a file or rebuild it from BigQuery if the file doesn't exist.

        :return: Dictionary containing the cache data
        """
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                return json.load(f)
        else:
            print(f"Cache file {self.filename} not found. Rebuilding from BigQuery...")
            return self.rebuild_from_bigquery()

    def rebuild_from_bigquery(self) -> Dict[str, Any]:
        """
        Rebuild the cache by querying BigQuery for existing submissions and comments.

        :return: Dictionary containing the rebuilt cache data
        """
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
        """
        Save the current cache to a file.

        :param cache: Optional cache data to save. If None, saves the current instance cache.
        """
        if cache is None:
            cache = self.cache
        with open(self.filename, 'w') as f:
            json.dump(cache, f, indent=2)

    def submission_exists(self, submission_id: str) -> bool:
        """
        Check if a submission exists in the cache.

        :param submission_id: ID of the submission to check
        :return: True if the submission exists in the cache, False otherwise
        """
        return submission_id in self.cache["submissions"]

    def get_submission_classification(self, submission_id: str) -> Dict[str, Any]:
        """
        Get the classification data for a submission from the cache.

        :param submission_id: ID of the submission
        :return: Classification data for the submission
        """
        if submission_id in self.cache["submissions"]:
            return self.cache["submissions"][submission_id]

    def comment_exists(self, comment_id: str) -> bool:
        """
        Check if a comment exists in the cache.

        :param comment_id: ID of the comment to check
        :return: True if the comment exists in the cache, False otherwise
        """
        return comment_id in self.cache["comments"]

    def add_submission(self, submission_id: str, data: Any):
        """
        Add a submission to the cache.

        :param submission_id: ID of the submission
        :param data: Data associated with the submission
        """
        self.cache["submissions"][submission_id] = data
        self.save()

    def add_comment(self, comment_id: str):
        """
        Add a comment to the cache.

        :param comment_id: ID of the comment
        """
        self.cache["comments"][comment_id] = {}
        self.save()

    def __len__(self) -> int:
        """
        Get the total number of submissions and comments in the cache.

        :return: Total count of cached items
        """
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
    """
    A class to store NLP processing results for text data.

    This class holds polarity, subjectivity, and entity information
    extracted from text using NLP techniques.
    """

    def __init__(self, polarity, subjectivity, entities):
        """
        Initialize NLPData instance.

        :param polarity: Sentiment polarity score
        :param subjectivity: Subjectivity score
        :param entities: List of extracted entities
        """
        self.polarity = polarity
        self.subjectivity = subjectivity
        self.entities = entities


# Accessing the reddit API details from my praw.ini file (which is in the same directory as the script)
# This information is kept private and not included in the repository
reddit = praw.Reddit("bot1")

# Choosing the subreddits to search for comments in
subreddits = reddit.subreddit("wellington+auckland+thetron+chch+dunedin+tauranga+newzealand+palmy+napier+newplymouth"
                              "+hawkesbay+NelsonNZ+queenstown")

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


# Function to retrieve all submission ids from bigquery
# @param data_type either 'comment' or 'submission'
def get_ids_from_bigquery(data_type):
    """
    Retrieve all submission or comment IDs from BigQuery.

    :param data_type: Either 'comment' or 'submission'
    :return: Set of IDs or None if the query fails
    """
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
    """
    Handle program exit by saving the submission cache.
    """
    print("Exiting")
    submission_cache.save()


def perform_nlp(doc):
    """
    Perform NLP analysis on the given text.

    :param doc: Input text to analyze
    :return: NLPData object containing polarity, subjectivity, and entities
    """
    processed_doc = nlp(doc)
    entities = []
    for ent in processed_doc.ents:
        if ent.label_ in labels:
            entities.append({'text': ent.text, 'label': ent.label_})

    return NLPData(processed_doc._.blob.polarity, processed_doc._.blob.subjectivity, entities)


def perform_classification(doc):
    """
    Perform zero-shot classification on the given text.

    :param doc: Input text to classify
    :return: Predicted topic label
    """
    hypothesis_template = 'This text is about {}.'
    prediction = classifier(doc, classification_topics, hypothesis_template=hypothesis_template, multi_label=True)
    if prediction and prediction['labels']:
        return prediction['labels'][0]


def insert_submission(submission, title_nlp, body_nlp, topic):
    """
    Insert a submission's data into BigQuery.

    :param submission: Reddit submission object
    :param title_nlp: NLPData for the submission title
    :param body_nlp: NLPData for the submission body
    :param topic: Classified topic for the submission
    """
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
    print("Inserted Submission: {}".format(submission.id))
    if errors:
        print(f"Errors inserting submission: {errors}")


def insert_comment(comment, comment_nlp, topic):
    """
    Insert a comment's data into BigQuery.

    :param comment: Reddit comment object
    :param comment_nlp: NLPData for the comment
    :param topic: Classified topic for the comment
    """
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
    print("Inserted Comment: {}".format(comment.id))
    if errors:
        print(f"Errors inserting comment: {errors}")


# Main loop to search for comments
while True:
    try:
        print("Starting Sentiment Analysis Bot")

        for submission in subreddits.new(limit=100000):

            if not submission_cache.submission_exists(submission.id):
                submission_title_processed = perform_nlp(submission.title)
                submission_body_processed = perform_nlp(submission.selftext)
                submission_text = "Post Title:" + submission.title + "Body:" + submission.selftext + "Comments:"
                submission.comments.replace_more(limit=None)
                iterator = 0
                for comment in submission.comments.list():
                    iterator += 1
                    if iterator > 10:
                        break
                    if comment.body and comment.body != "":
                        submission_text += comment.body + ". Comment:"

                classification_prediction = perform_classification(submission_text)
                insert_submission(submission, submission_title_processed, submission_body_processed,
                                  classification_prediction)

                # Update cache
                submission_cache.add_submission(submission.id, classification_prediction)

            else:
                classification_prediction = submission_cache.get_submission_classification(submission.id)

            submission.comments.replace_more(limit=None)
            for comment in submission.comments.list():
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

        print("Installing spaCy")
        download("en_core_web_lg")

    # Exiting cleanly when the program is interrupted by the user
    except KeyboardInterrupt:
        print("Stopped by user")
        exit_handler()
        break

    except Exception as e:
        print("Unknown Program Failure")
        print(traceback.print_exc())
        exit_handler()
        break
    except prawcore.exceptions.ServerError as e:
        print("Server Error")
        time.sleep(200)
