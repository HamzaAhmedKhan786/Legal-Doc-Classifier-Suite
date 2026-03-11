import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

class LegalTextPreprocessor:
    def __init__(self):
        # We define common legal "noise" patterns
        self.case_pattern = re.compile(r'case no\.:?\s*\d+', re.IGNORECASE)
        self.date_pattern = re.compile(r'\d{1,2}/\d{1,2}/\d{2,4}')

    def clean_text(self, text):
        """Removes noise, punctuation, and standardizes legal text."""
        if not isinstance(text, str):
            return ""
        
        # 1. Lowercasing
        text = text.lower()
        
        # 2. Remove Case Numbers and Dates (Generic noise)
        text = self.case_pattern.sub('', text)
        text = self.date_pattern.sub('', text)
        
        # 3. Remove Punctuation and special characters
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # 4. Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def apply_preprocessing(self, series):
        """Applies cleaning to a full pandas column."""
        return series.apply(self.clean_text)