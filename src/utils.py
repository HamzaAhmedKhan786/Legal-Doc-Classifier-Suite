import joblib
import re

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

# Changed to singular and made it generic to save ANY object (model or vectorizer)
def save_artifact(obj, filepath):
    """Saves a python object (model/vectorizer) to the specified path."""
    joblib.dump(obj, filepath)

def load_artifact(filepath):
    """Loads a python object from the specified path."""
    return joblib.load(filepath)