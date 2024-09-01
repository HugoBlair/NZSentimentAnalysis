import unittest
from unittest.mock import Mock, patch
from datetime import datetime, timezone
import json

# Import the necessary classes and functions from your main script
from redditScraper import NLPData, SubmissionCache, perform_nlp, perform_classification


class TestRedditSentimentAnalysis(unittest.TestCase):

    def setUp(self):
        # Set up any necessary test data or mocks
        self.mock_bigquery_client = Mock()
        self.mock_dataset_ref = Mock()

    def test_nlp_data_creation(self):
        nlp_data = NLPData(polarity=0.5, subjectivity=0.7, entities=[{'text': 'New Zealand', 'label': 'GPE'}])
        self.assertEqual(nlp_data.polarity, 0.5)
        self.assertEqual(nlp_data.subjectivity, 0.7)
        self.assertEqual(len(nlp_data.entities), 1)
        self.assertEqual(nlp_data.entities[0]['text'], 'New Zealand')

    @patch('os.path.exists')
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data='{"submissions": {}, "comments": {}}')
    def test_submission_cache_load(self, mock_open, mock_exists):
        mock_exists.return_value = True
        cache = SubmissionCache('test_cache.json', self.mock_bigquery_client, self.mock_dataset_ref)
        self.assertEqual(len(cache.cache['submissions']), 0)
        self.assertEqual(len(cache.cache['comments']), 0)

    @patch('redditScraper.nlp')
    def test_perform_nlp(self, mock_nlp):
        mock_doc = Mock()
        mock_doc._.blob.polarity = 0.5
        mock_doc._.blob.subjectivity = 0.7
        mock_doc.ents = [Mock(text='New Zealand', label_='GPE')]
        mock_nlp.return_value = mock_doc

        result = perform_nlp("Test document about New Zealand")
        self.assertEqual(result.polarity, 0.5)
        self.assertEqual(result.subjectivity, 0.7)
        self.assertEqual(len(result.entities), 1)
        self.assertEqual(result.entities[0]['text'], 'New Zealand')
        self.assertEqual(result.entities[0]['label'], 'GPE')

    @patch('redditScraper.classifier')
    def test_perform_classification(self, mock_classifier):
        mock_classifier.return_value = {'labels': ['Politics']}
        result = perform_classification("This is a political document")
        self.assertEqual(result, 'Politics')


if __name__ == '__main__':
    unittest.main()
