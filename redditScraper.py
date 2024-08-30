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

#Load comments replied to from file
if not os.path.isfile("comments_replied_to.txt"):
    print("Creating comments_replied_to.txt")
    comments_replied_to = []
else:
    with open("comments_replied_to.txt", "r") as f:
        print("Loading comments_replied_to.txt")
        comments_replied_to = f.read()
        comments_replied_to = comments_replied_to.split("\n")
        comments_replied_to = list(filter(None, comments_replied_to))

#Function to save comments replied to
def exit_handler():
    print("Exiting")
    with open("comments_replied_to.txt", "w") as f:
        for comment_id in comments_replied_to:
            f.write(comment_id + "\n")
    print("Saved comments replied to")

#Main loop to search for comments
try:
    print("Starting Sentiment Analysis Bot")
    bot_username = reddit.user.me().name
    print("Logged in as " + bot_username)

    #Loading spaCy
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe('spacytextblob')
    print("Initalized spaCy")

    for submission in subreddits.hot(limit=3):
        doc = nlp(submission.title)
        print(submission.title)
        submission_title_polarity = doc._.blob.polarity
        submission_title_subjectivity = doc._.blob.subjectivity
        doc = nlp(submission.selftext)
        submission_body_polarity = doc._.blob.polarity
        submission_body_subjectivity = doc._.blob.subjectivity

        for comment in submission.comments:
            print("Found new comment")
            if comment.author.name != bot_username:
                if comment.id not in comments_replied_to:
                    doc = nlp(comment.body)
                else:
                    print("Already replied to comment")
                    print(comment.body)
                    comments_replied_to.append(comment.id)






#Catching case where spaCy's model in not installed
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    #Exiting cleanly when the program is interrupted by the user
except KeyboardInterrupt:
    print("Stopped by user")
    exit_handler()
#Exiting cleanly when the program is interrupted by the user
except Exception:
    print("Unknown Program Failure")
    exit_handler()
