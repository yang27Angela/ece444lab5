from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Create Flask app instance
application = Flask(__name__)

@application.route("/")
def index():
    return "Your Flask App Works! V1.0"

# Load the model and vectorizer
def load_model():
    global model, vectorizer
    try:
        # Load the model and vectorizer from the files (replace with actual paths if necessary)
        with open('basic_classifier.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('count_vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        print("Model and vectorizer loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

# Load the model and vectorizer when the application starts
load_model()

@application.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json()
        text = data['text']
        
        # Transform the text using the loaded vectorizer
        transformed_text = vectorizer.transform([text])
        
        # Make a prediction using the loaded model
        prediction = model.predict(transformed_text)

        # Map prediction result to numeric values (assuming 'FAKE' or 'REAL' are the output labels)
        prediction_mapping = {
            'FAKE': 1,
            'REAL': 0
        }
        result = prediction_mapping.get(prediction[0], "Unknown")  # Convert to numeric or handle unexpected labels
        
        return jsonify({'prediction': result})
    except Exception as e:
        # Return an error message if something goes wrong
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    application.run(port=5000, debug=True)
