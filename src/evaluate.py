import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Generates and saves a heatmap of the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('Actual Category')
    plt.xlabel('Predicted Category')
    
    # Save the plot to the artifacts folder
    plt.savefig(f'artifacts/{model_name}_confusion_matrix.png')
    plt.close()

def evaluate_model(model, X_test, y_test, model_name):
    predictions = model.predict(X_test)
    print(f"--- Evaluation for {model_name} ---")
    print(classification_report(y_test, predictions))
    
    # Visual evaluation
    plot_confusion_matrix(y_test, predictions, model_name)