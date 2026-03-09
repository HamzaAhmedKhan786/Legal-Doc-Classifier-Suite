import joblib
import re

# Utility functions for text preprocessing and model persistence
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

# Save and load model and vectorizer using joblib
def save_artifacts(model, vectorizer, model_path='model.joblib', vectorizer_path='vectorizer.joblib'):
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

# Load model and vectorizer from disk
def load_artifacts(model_path='model.joblib', vectorizer_path='vectorizer.joblib'):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer