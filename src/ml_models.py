import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import joblib
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, f1_score

from src.nlp_features import FeatureExtractor, load_data_splits

class MultiTargetMLClassifier:
    """Wrapper to train and predict multi-target labels using traditional ML algorithms."""
    def __init__(self, model_type="logistic_regression"):
        self.model_type = model_type
        self.targets = ["category", "priority", "sentiment", "department"]
        self.models = {}
        
    def _create_model(self):
        if self.model_type == "naive_bayes":
            return MultinomialNB(alpha=0.5)
        elif self.model_type == "logistic_regression":
            return LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        elif self.model_type == "svm":
            return LinearSVC(C=1.0, random_state=42, dual=False)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(self, X_tfidf, y_dict):
        for target in self.targets:
            clf = self._create_model()
            clf.fit(X_tfidf, y_dict[target])
            self.models[target] = clf

    def predict(self, X_tfidf):
        predictions = {}
        for target, clf in self.models.items():
            predictions[target] = clf.predict(X_tfidf)
        return predictions

    def predict_proba(self, X_tfidf):
        probabilities = {}
        for target, clf in self.models.items():
            if hasattr(clf, "predict_proba"):
                probabilities[target] = clf.predict_proba(X_tfidf)
            elif hasattr(clf, "decision_function"):
                df = clf.decision_function(X_tfidf)
                # Softmax conversion of decision function scores for SVM
                exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
                probabilities[target] = exp_df / np.sum(exp_df, axis=1, keepdims=True)
        return probabilities

    def save(self, model_dir="models/ml"):
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, f"{self.model_type}_model.pkl")
        joblib.dump(self, path)
        print(f"Saved {self.model_type} model to {path}")

    @staticmethod
    def load(filepath):
        return joblib.load(filepath)

def train_and_evaluate_ml_baselines():
    train_df, val_df, test_df = load_data_splits()
    
    fe = FeatureExtractor(max_features=5000, ngram_range=(1, 2))
    fe.fit_labels(train_df)
    
    X_train = fe.fit_transform_tfidf(train_df["ticket_text"])
    X_val = fe.transform_tfidf(val_df["ticket_text"])
    X_test = fe.transform_tfidf(test_df["ticket_text"])
    
    y_train = fe.encode_labels(train_df)
    y_val = fe.encode_labels(val_df)
    y_test = fe.encode_labels(test_df)
    
    fe.save("models/feature_extractor.pkl")
    
    algorithms = ["naive_bayes", "logistic_regression", "svm"]
    results = {}
    
    print("\n=================== ML BASELINE TRAINING ===================")
    for algo in algorithms:
        print(f"\nTraining {algo.upper()}...")
        classifier = MultiTargetMLClassifier(model_type=algo)
        classifier.fit(X_train, y_train)
        classifier.save("models/ml")
        
        preds_test = classifier.predict(X_test)
        
        algo_results = {}
        for target in ["category", "priority", "sentiment", "department"]:
            acc = accuracy_score(y_test[target], preds_test[target])
            f1 = f1_score(y_test[target], preds_test[target], average="weighted")
            algo_results[target] = {"accuracy": acc, "f1_score": f1}
            print(f"  [{target.capitalize():10s}] Test Accuracy: {acc*100:.2f}% | F1-Score: {f1:.4f}")
            
        results[algo] = algo_results
        
    return results

if __name__ == "__main__":
    train_and_evaluate_ml_baselines()
