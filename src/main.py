from dataloader import load_legal_data, prepare_splits
from preprocess import LegalTextPreprocessor
from train import train_vectorizer, get_models
from evaluate import evaluate_model
from utils import save_artifact

def run_pipeline():
    df = load_legal_data('legal_text_classification.csv')
    
    # NEW STEP: Cleaning the data before splitting
    preprocessor = LegalTextPreprocessor()
    print("Cleaning legal text data...")
    df['case_text'] = preprocessor.apply_preprocessing(df['case_text'])
    
    X_train_raw, X_test_raw, y_train, y_test = prepare_splits(df)

    # 2. Vectorize
    print("Vectorizing text...")
    X_train, vectorizer = train_vectorizer(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    save_artifact(vectorizer, "vectorizer.joblib")

    # 3. Train & Evaluate
    models = get_models()
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        evaluate_model(model, X_test, y_test, name)
        save_artifact(model, f"{name}_model.joblib")

if __name__ == "__main__":
    run_pipeline()