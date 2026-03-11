from sklearn.metrics import classification_report, accuracy_score

def evaluate_model(model, X_test, y_test, model_name):
    predictions = model.predict(X_test)
    print(f"--- Evaluation for {model_name} ---")
    print(classification_report(y_test, predictions))
    return accuracy_score(y_test, predictions)