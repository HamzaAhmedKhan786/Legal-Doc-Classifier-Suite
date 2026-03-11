from utils import load_artifact

class LegalClassifier:
    def __init__(self, model_type="SVC"):
        self.vectorizer = load_artifact("vectorizer.joblib")
        self.model = load_artifact(f"{model_type}_model.joblib")

    def predict(self, text):
        processed_text = self.vectorizer.transform([text])
        prediction = self.model.predict(processed_text)
        return prediction[0]

if __name__ == "__main__":
    # Quick test
    clf = LegalClassifier(model_type="SVC")
    sample_text = "The court finds the defendant liable for breach of contract..."
    result = clf.predict(sample_text)
    print(f"Result: {result}")