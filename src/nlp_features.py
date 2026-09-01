import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib

from src.preprocessing import clean_text

class FeatureExtractor:
    """Handles TF-IDF & Label Encoding for ML & DL experiments."""
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            preprocessor=clean_text
        )
        self.label_encoders = {
            "category": LabelEncoder(),
            "priority": LabelEncoder(),
            "sentiment": LabelEncoder(),
            "department": LabelEncoder()
        }
        
    def fit_labels(self, df):
        for col in self.label_encoders:
            if col in df.columns:
                self.label_encoders[col].fit(df[col])

    def encode_labels(self, df):
        encoded = {}
        for col, le in self.label_encoders.items():
            if col in df.columns:
                encoded[col] = le.transform(df[col])
        return encoded

    def decode_labels(self, col, indices):
        return self.label_encoders[col].inverse_transform(indices)

    def fit_transform_tfidf(self, train_texts):
        return self.tfidf.fit_transform(train_texts)

    def transform_tfidf(self, texts):
        return self.tfidf.transform(texts)

    def save(self, filepath="models/feature_extractor.pkl"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Saved FeatureExtractor to {filepath}")

    @staticmethod
    def load(filepath="models/feature_extractor.pkl"):
        return joblib.load(filepath)

def load_data_splits(data_dir="data"):
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    return train_df, val_df, test_df

if __name__ == "__main__":
    train_df, val_df, test_df = load_data_splits()
    fe = FeatureExtractor()
    fe.fit_labels(train_df)
    X_train_tfidf = fe.fit_transform_tfidf(train_df["ticket_text"])
    print("TF-IDF Matrix Shape:", X_train_tfidf.shape)
    for k, le in fe.label_encoders.items():
        print(f"Classes for {k}: {le.classes_}")
