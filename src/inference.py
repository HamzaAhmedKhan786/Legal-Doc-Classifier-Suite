from utils import load_artifact
import os

class LegalClassifier:
    def __init__(self, model_type="SVC"):
        # Points to the artifacts folder
        self.vectorizer = load_artifact(os.path.join('artifacts', "vectorizer.joblib"))
        self.model = load_artifact(os.path.join('artifacts', f"{model_type}_model.joblib"))

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