from datetime import datetime
import os
import praw

import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

#Accessing the reddit API details from my praw.ini file (which is in the same directory as the script)
#This information is kept private and not included in the repository
reddit = praw.Reddit('sentimentAnalysisBot')

#Subreddit list
subreddits = {
    "wellington",
    "auckland",
    "thetron",
    "chch",
    "dunedin",
    "tauranga",
    "newzealand"
}

#Choosing the subreddits to search for comments in
subreddit = reddit.subreddit()

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

    #Initializing spaCy + spaCy TextBlob
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('spacytextblob')

    for subreddit in subreddits:

        for submission in reddit.subreddit(subreddit):
            doc = nlp(submission.body)
            submission_polarity = doc._.blob.polarity
            print("Submission polarity: " + str(submission_polarity))
            for comment in submission:
                print("Found new comment")
                if comment.author.name != bot_username:
                    if comment.id not in comments_replied_to:
                        doc = nlp(comment.body)
                    else:
                        print("Already replied to comment")
                        print(comment.body)





except KeyboardInterrupt:
    print("Stopped by user")
    exit_handler()