from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

def train_vectorizer(train_text):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1,2))
    X_vec = vectorizer.fit_transform(train_text)
    return X_vec, vectorizer

def get_models():
    return {
        "Naive_Bayes": MultinomialNB(),
        "SVC": LinearSVC(max_iter=2000),
        "Random_Forest": RandomForestClassifier(n_estimators=100)
    }