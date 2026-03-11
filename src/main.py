import os
from src.dataloader import load_legal_data, prepare_splits
from src.preprocess import LegalTextPreprocessor
from src.train import train_vectorizer, get_models
from src.evaluate import evaluate_model
from src.utils import save_artifact

def run_pipeline():
    # Define paths relative to the project root
    data_path = os.path.join('data', 'legal_text_classification.csv')
    artifact_dir = 'artifacts'
    
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    # 1. Load
    print(f"Loading data from {data_path}...")
    df = load_legal_data(data_path)
    
    # 2. Preprocess
    preprocessor = LegalTextPreprocessor()
    df['case_text'] = preprocessor.apply_preprocessing(df['case_text'])
    
    # 3. Split & Vectorize
    X_train_raw, X_test_raw, y_train, y_test = prepare_splits(df)
    X_train, vectorizer = train_vectorizer(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    
    save_artifact(vectorizer, os.path.join(artifact_dir, "vectorizer.joblib"))

    # 4. Train & Evaluate
    models = get_models()
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        evaluate_model(model, X_test, y_test, name)
        save_artifact(model, os.path.join(artifact_dir, f"{name}_model.joblib"))

if __name__ == "__main__":
    run_pipeline()