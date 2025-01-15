# NZSentimentAnalysis

This is a comprehensive sentiment analysis and topic classification system for New Zealand-focused subreddits, combining natural language processing, machine learning, and data visualization to analyze discussions across various New Zealand communities on Reddit.

## Access the website here:
https://hugoblair-nzsentimentanalysis-nzsentimentanalysiswebsite-dfeviv.streamlit.app/

## Project Overview
This project consists of two main components:

- A data collection and analysis pipeline that processes submissions and comments scraped from reddit
- An interactive Streamlit dashboard for visualizing the analyzed data

The system analyzes posts from multiple New Zealand city and regional subreddits, including:

- r/wellington
- r/auckland
- r/thetron
- r/chch
- r/dunedin
- r/tauranga
- r/newzealand
- r/palmy
- r/napier
- r/newplymouth
- r/hawkesbay
- r/NelsonNZ
- r/queenstown

## Use case
This tool enables researchers, policymakers, and community leaders to understand regional variations in public discourse across New Zealand. The system can identify trending topics and sentiment patterns specific to different cities and regions, revealing how different communities react to local and national issues. For example, housing discussions might show higher negative sentiment in Auckland compared to smaller cities, or environmental topics might dominate Queenstown's discourse more than urban centers. The entity recognition system can track which organizations, politicians, and locations are most frequently discussed in different regions, while analysis of trends over time could reveal how community concerns evolve. This data could be valuable for local governments understanding community priorities, businesses gauging regional market sentiment, or researchers studying the unique characteristics of New Zealand's online communities.

### Disclaimer:
This data is sourced entirely from reddit. The regional subreddits aren't entirely representative of the communities they are about for many reasons such as reddit users being disproportionately younger, tech savvy, and left leaning than the general populations. The data is gathered over a period of a couple of months (11/24), so it reflects the topics trending at the time. The data also has the risk of being misclassified. The data in it's current form is best used for making comparisons between cities, not drawing absolute conclusions about an individual city.
 
## Features
Data Collection & Analysis

- Sentiment Analysis: Measures the polarity and subjectivity of submissions and comments
- Entity Recognition: Identifies and categorizes named entities like locations, organizations, and people
- Topic Classification: Uses zero-shot classification to categorize discussions into predefined topics
- Caching System: Implements efficient data processing with a local cache to prevent reprocessing
- BigQuery Integration: Stores processed data in Google BigQuery for scalable data management

## Data Visualization Dashboard:

### Interactive Filters: 
Date range selection and minimum submission thresholds

### Topic Analysis:

- Topic distribution across subreddits
- Topic trends over time
- Topic heatmap visualization

### Sentiment Analysis:

- Sentiment vs. subjectivity scatter plots
- Sentiment distribution across communities


### Entity Analysis:

- Frequency analysis of mentioned entities
- Entity type categorization
- Searchable entity visualization


## Work to be done:
- Hosting script on an ec2 instance to automate scraping.
- Fleshing out website with more tools and features.


## Data Structure
The project uses two main BigQuery tables:

### Submissions Table

- Submission ID
- Title sentiment (polarity and subjectivity)
- Body sentiment (polarity and subjectivity)
- Timestamp
- Subreddit
- Named entities (from title and body)
- Topic classification

### Comments Table

- Comment ID
- Submission ID
- Sentiment metrics
- Timestamp
- Named entities
- Inherited topic classification

## Classification Categories and Entities
### Topic Categories
The system classifies content into the following categories:

- Politics
- Economics
- Sports
- Technology
- Environment
- Health
- Education
- Culture
- Social Issues
- Transport
- Tourism
- Agriculture
- Foreign Affairs
- Housing
- Justice and Law
- Indigenous Affairs
- Other/Unknown

### Entity Recognition
The system recognizes various types of named entities:

- EVENT: Notable events and happenings
- FAC: Buildings, airports, highways, bridges
- GPE: Countries, cities, states
- LAW: Named laws and regulations
- LOC: Non-GPE locations, mountain ranges, water bodies
- NORP: Nationalities, religious, or political groups
- ORG: Companies, agencies, institutions
- PERSON: People, including fictional
- PRODUCT: Products
- WORK_OF_ART: Titles of books, songs, etc.

## Example Of Website For Reference If Website Goes Down
![image](https://github.com/user-attachments/assets/1e945a8d-f901-44fa-9f30-7a796a4ab851)
