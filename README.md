# ⚖️ Legal Document Classifier Suite

An end-to-end modular pipeline for high-performance classification of legal documents. This project compares classical supervised learning models (SVC, Random Forest, Naive Bayes) to establish an efficient baseline for legal-tech workflows.

## 🚀 Overview
In production legal environments (like the ones I've built for Jurasoft), processing millions of documents with LLMs is often cost-prohibitive. This project demonstrates a **"First-Pass Filter"** approach: using lightweight, highly optimized traditional ML models to categorize 90% of documents at near-zero cost, reserving GPU-heavy models (like Llama 3 or LeoLM) only for complex edge cases.

## 🛠️ Tech Stack
- **Languages:** Python
- **ML Framework:** Scikit-learn
- **NLP Techniques:** TF-IDF Vectorization, N-grams (1,2), Tokenization
- **Models:** Linear SVC, Random Forest, Multinomial Naive Bayes

## 📊 Dataset
This project utilizes the [Legal Text Classification Dataset](https://www.kaggle.com/datasets/amohankumar/legal-text-classification-dataset) from Kaggle, containing thousands of legal cases labeled by outcome and domain.

## 🏗️ Modular Architecture
The system is built using a production-grade modular structure:
- `dataloader.py`: Handles data ingestion and cleaning logic.
- `train.py`: Defines model architectures and hyperparameter configurations.
- `evaluate.py`: Generates comprehensive classification reports and F1-score analysis.
- `inference.py`: A standalone class for real-time prediction on unseen legal text.
- `utils.py`: Contains text-cleansing regex and artifact serialization helpers.

## 📈 Model Comparison Rationale
| Model | Rationale | Performance Role |
| :--- | :--- | :--- |
| **Naive Bayes** | Probability-based baseline. | Fast sorting of high-volume document streams. |
| **SVC** | High-dimensional hyperplane separation. | The "Precision Specialist" for overlapping legal classes. |
| **Random Forest** | Ensemble decision trees. | Robustness against outliers in unformatted legal text. |

## ⚙️ Installation & Usage
1. **Clone the repo:**
   ```bash
   git clone [https://github.com/HamzaAhmedKhan786/Legal-Doc-Classifier-Suite.git](https://github.com/HamzaAhmedKhan786/Legal-Doc-Classifier-Suite.git)
