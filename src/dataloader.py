import pandas as pd
from sklearn.model_selection import train_test_split

# Data loading function
def load_legal_data(filepath):
    """"Loads and cleans the legal case dataset from a CSV file."""
    df = pd.read_csv(filepath)
    # Ensure no empty text or labels
    df = df.dropna(subset=['case_text', 'case_outcome'])
    return df
    
# Function to split the dataset into training and testing sets
# Change this line in dataloader.py
def prepare_splits(df, test_size=0.2, random_state=42):
    # Split the dataset into training and testing sets
    X = df['case_text']
    y = df['case_outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test