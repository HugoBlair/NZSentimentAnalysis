import os
import praw
import re
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import requests

#Accessing the reddit API details from my praw.ini file (which is in the same directory as the script)
#This information is kept private and not included in the repository
reddit = praw.Reddit("bot1")

#Choosing the subreddits to search for comments in
subreddits = reddit.subreddit("wellington+auckland+thetron+chch+dunedin+tauranga+newzealand")

#List of acceptable labels for NLP tagging
labels = {
    'EVENT', 'FAC', 'GPE', 'LAW', 'LOC', 'NORP', 'ORG', 'PERSON', 'PRODUCT', 'WORK_OF_ART'
}

#Load comments replied to from file
if not os.path.isfile("comments_seen.txt"):
    print("Creating comments_seen.txt")
    comments_seen = []
    submissions_seen = []
else:
    with open("comments_seen.txt", "r") as f:
        print("Loading comments_seen.txt")
        comments_seen = f.read()
        comments_seen = comments_seen.split("\n")
        comments_seen = list(filter(None, comments_seen))


#Function to save comments replied to
def exit_handler():
    print("Exiting")
    with open("comments_seen.txt", "w") as f:
        for comment_id in comments_seen:
            f.write(comment_id + "\n")
    print("Saved comments replied to")


def perform_nlp(doc):
    for ent in doc.ents:
        print(ent.text, ent.label_)
        if ent.label_ in labels:
            nlp_labels = nlp(ent.text)

    return nlp_data(doc._.blob.polarity,doc._.blob.subjectivity, nlp_labels)



#Main loop to search for comments
try:
    print("Starting Sentiment Analysis Bot")
    #Loading spaCy
    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe('spacytextblob')

    print("Initalized spaCy")

    for submission in subreddits.hot(limit=3):
        if submission not in submissions_seen:
            submission_title_processed =perform_nlp(submission.title)
            submission_body_processed = perform_nlp(submission.selftext)
        else:
            submissions_seen.append(submission.title)

        for comment in submission.comments:
            #print("Found new comment")
            if comment.id not in comments_seen:
                comment_processed = perform_nlp(comment)

            else:
                print("Already replied to comment")
                print(comment.body)
                comments_seen.append(comment.id)


#Catching case where spaCy's model in not installed
except OSError:
    from spacy.cli import download

    download("en_core_web_lg")
    #Exiting cleanly when the program is interrupted by the user
except KeyboardInterrupt:
    print("Stopped by user")
    exit_handler()
#Exiting cleanly when the program is interrupted by the user
except Exception as e:
    print("Unknown Program Failure")
    print(e)
    exit_handler()

class nlp_data:
    def __init__(self, polarity, subjectivity,labels):

        self.polarity = polarity
        self.subjectivity = subjectivity
        self.labels = labels

