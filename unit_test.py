import unittest
import requests
from flask import json
import time
import pandas as pd
import matplotlib.pyplot as plt

class FakeNewsFlaskTest(unittest.TestCase):

    def setUp(self):
        # Base URL of the deployed Flask application
        self.url = "http://lab5-env.eba-cwskmhnk.us-east-2.elasticbeanstalk.com"

    def test_real_news_prediction(self):
        # Test case for predicting real news
        response = requests.post(self.url + '/predict',
                                 data=json.dumps({'text': 'Apple is fruit'}),
                                 headers={'Content-Type': 'application/json'})
        data = response.json()
        self.assertEqual(data['prediction'], True, "Prediction should be 'True' for real news text.")

        response = requests.post(self.url + '/predict',
                                 data=json.dumps({'text': 'Lemon is fruit'}),
                                 headers={'Content-Type': 'application/json'})
        data = response.json()
        self.assertEqual(data['prediction'], True, "Prediction should be 'True' for real news text.")

        response = requests.post(self.url + '/predict',
                                 data=json.dumps({'text': 'one'}),
                                 headers={'Content-Type': 'application/json'})
        data = response.json()
        self.assertEqual(data['prediction'], True, "Prediction should be 'True' for real news text.")

    def test_fake_news_prediction(self):
        # Test case for predicting fake news
        response = requests.post(self.url + '/predict',
                                 data=json.dumps({'text': 'one day is one hour'}),
                                 headers={'Content-Type': 'application/json'})
        data = response.json()
        self.assertEqual(data['prediction'], False, "Prediction should be 'False' for fake news text.")

        response = requests.post(self.url + '/predict',
                                 data=json.dumps({'text': 'one hour is one second'}),
                                 headers={'Content-Type': 'application/json'})
        data = response.json()
        self.assertEqual(data['prediction'], False, "Prediction should be 'False' for fake news text.")

    def test_invalid_json(self):
        # Test case for an invalid JSON (should ideally return an error)
        response = requests.post(self.url + '/predict',
                                 data=123,
                                 headers={'Content-Type': 'application/json'})
        self.assertEqual(response.status_code, 500, "Response should return a 500 status for invalid JSON.")  # Update as per your server's handling of invalid JSON

    def test_100_calls(self):
    # Test case for making 100 API calls and recording response times
        response_times = {'Real News': [], 'Fake News': []}
        test_cases = [
            {'text': 'Apple is a fruit'},  # Real news
            {'text': 'One day is one hour'}  # Fake news
        ]

    # Make 100 requests for each test case
        for _ in range(100):
            for i, (label, test) in enumerate(zip(response_times.keys(), test_cases)):
                start_time = time.time()
                esponse = requests.post(self.url + '/predict',
                                     data=json.dumps(test),
                                     headers={'Content-Type': 'application/json'})
                end_time = time.time()
                response_times[label].append(end_time - start_time)

    # Convert the response times to a DataFrame and save them to a CSV file
        df = pd.DataFrame(response_times)
        df.to_csv('response_times.csv', index=False, header=True)

    # Plot the response times using a boxplot
        plt.figure()
        df.boxplot()
        plt.title("API Response Times for 100 Calls")
        plt.ylabel("Time (seconds)")
        plt.show()

    # Verify that the average response time across all requests is less than 1 second
        avg_response_time = df.mean().mean()  # Calculate average across all columns and rows
        self.assertLess(avg_response_time, 1.0, "Average response time should be less than 1 second.")


if __name__ == '__main__':
    unittest.main()
