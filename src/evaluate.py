import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import SimpleTokenizer, clean_text
from src.nlp_features import FeatureExtractor, load_data_splits
from src.ml_models import MultiTargetMLClassifier
from src.dl_models import MultiTaskBiLSTM

# Styling settings for professional plots
sns.set_theme(style="darkgrid")
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

def generate_wordcloud(texts, save_path=os.path.join(FIGURES_DIR, "wordcloud_tickets.png")):
    cleaned_corpus = " ".join([clean_text(t) for t in texts])
    wc = WordCloud(width=1200, height=600, background_color="#0f172a", colormap="viridis", max_words=150).generate(cleaned_corpus)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Most Frequent Terms in Customer Support Tickets", fontsize=16, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Generated WordCloud -> {save_path}")


def generate_ngram_chart(texts, save_path=os.path.join(FIGURES_DIR, "top_tfidf_ngrams.png")):
    vec = TfidfVectorizer(max_features=15, ngram_range=(1, 2), preprocessor=clean_text)
    matrix = vec.fit_transform(texts)
    scores = np.asarray(matrix.mean(axis=0)).ravel()
    features = np.array(vec.get_feature_names_out())
    
    df_ngram = pd.DataFrame({"ngram": features, "importance": scores}).sort_values(by="importance", ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(df_ngram["ngram"], df_ngram["importance"], color="#3b82f6")
    plt.xlabel("Mean TF-IDF Score", fontsize=12, fontweight="bold")
    plt.title("Top 15 Most Important N-Grams (TF-IDF)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Generated N-Gram Chart -> {save_path}")


def plot_confusion_matrix(y_true, y_pred, labels, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
    plt.ylabel("True Label", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Confusion Matrix -> {save_path}")


def plot_model_comparison(results_summary, save_path_acc=os.path.join(FIGURES_DIR, "model_comparison_accuracy.png"),
                           save_path_f1=os.path.join(FIGURES_DIR, "model_comparison_f1.png")):
    models = list(results_summary.keys())
    cat_accs = [results_summary[m]["category"]["accuracy"] * 100 for m in models]
    cat_f1s = [results_summary[m]["category"]["f1_score"] for m in models]
    
    # Accuracy Bar Chart
    plt.figure(figsize=(10, 5))
    bars = plt.bar(models, cat_accs, color="#6366f1", width=0.5)
    plt.title("Category Classification Accuracy Across Models", fontsize=14, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    plt.ylim(80, 105)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.2f}%", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path_acc, bbox_inches="tight")
    plt.close()
    
    # F1 Score Bar Chart
    plt.figure(figsize=(10, 5))
    bars = plt.bar(models, cat_f1s, color="#10b981", width=0.5)
    plt.title("Category Classification Weighted F1-Score Across Models", fontsize=14, fontweight="bold")
    plt.ylabel("Weighted F1 Score", fontsize=12, fontweight="bold")
    plt.ylim(0.80, 1.05)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.4f}", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path_f1, bbox_inches="tight")
    plt.close()
    print(f"Saved Comparison Charts -> {save_path_acc}, {save_path_f1}")


def plot_dl_curves(history_path="models/dl/history.json"):
    if not os.path.exists(history_path):
        return
    with open(history_path, "r") as f:
        hist = json.load(f)
        
    mt_hist = hist.get("multi_task", {})
    epochs = list(range(1, len(mt_hist.get("train_loss", [])) + 1))
    
    # Loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, mt_hist.get("train_loss", []), label="Train Loss", color="#ef4444", linewidth=2.5, marker="o")
    plt.plot(epochs, mt_hist.get("val_loss", []), label="Val Loss", color="#3b82f6", linewidth=2.5, marker="s")
    plt.title("Multi-Task BiLSTM Training & Validation Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12, fontweight="bold")
    plt.ylabel("Loss", fontsize=12, fontweight="bold")
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "training_loss_curves.png"), bbox_inches="tight")
    plt.close()
    
    # Accuracy curves
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [a*100 for a in mt_hist.get("val_category_acc", [])], label="Category Acc", color="#10b981", linewidth=2, marker="o")
    plt.plot(epochs, [a*100 for a in mt_hist.get("val_priority_acc", [])], label="Priority Acc", color="#f59e0b", linewidth=2, marker="^")
    plt.plot(epochs, [a*100 for a in mt_hist.get("val_sentiment_acc", [])], label="Sentiment Acc", color="#8b5cf6", linewidth=2, marker="s")
    plt.title("Multi-Task BiLSTM Validation Accuracy Curves", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "training_accuracy_curves.png"), bbox_inches="tight")
    plt.close()
    print("Saved DL Training Curves.")


def run_comprehensive_evaluation():
    train_df, val_df, test_df = load_data_splits()
    
    # Generate dataset plots
    generate_wordcloud(train_df["ticket_text"])
    generate_ngram_chart(train_df["ticket_text"])
    
    fe = FeatureExtractor.load("models/feature_extractor.pkl")
    X_test_tfidf = fe.transform_tfidf(test_df["ticket_text"])
    y_test_encoded = fe.encode_labels(test_df)
    
    results_summary = {}
    
    # Evaluate Baseline ML Models
    for algo in ["naive_bayes", "logistic_regression", "svm"]:
        model_path = os.path.join("models/ml", f"{algo}_model.pkl")
        if os.path.exists(model_path):
            clf = MultiTargetMLClassifier.load(model_path)
            preds = clf.predict(X_test_tfidf)
            
            algo_res = {}
            for target in ["category", "priority", "sentiment", "department"]:
                acc = accuracy_score(y_test_encoded[target], preds[target])
                p, r, f1, _ = precision_recall_fscore_support(y_test_encoded[target], preds[target], average="weighted")
                algo_res[target] = {"accuracy": float(acc), "precision": float(p), "recall": float(r), "f1_score": float(f1)}
            results_summary[algo] = algo_res

    # Evaluate Multi-Task BiLSTM Model
    mt_model_path = "models/dl/multitask_bilstm.pt"
    tokenizer_path = "models/dl/tokenizer.pkl"
    
    if os.path.exists(mt_model_path) and os.path.exists(tokenizer_path):
        tokenizer = joblib.load(tokenizer_path)
        seqs = tokenizer.texts_to_sequences(test_df["ticket_text"], max_len=50)
        x_tensor = torch.tensor(seqs, dtype=torch.long)
        
        mt_model = MultiTaskBiLSTM(
            vocab_size=len(tokenizer.word2idx),
            num_categories=len(fe.label_encoders["category"].classes_),
            num_priorities=len(fe.label_encoders["priority"].classes_),
            num_sentiments=len(fe.label_encoders["sentiment"].classes_)
        )
        mt_model.load_state_dict(torch.load(mt_model_path, map_location=torch.device('cpu')))
        mt_model.eval()
        
        with torch.no_grad():
            outputs = mt_model(x_tensor)
            
        mt_preds = {}
        for task in ["category", "priority", "sentiment"]:
            mt_preds[task] = torch.argmax(outputs[task], dim=1).numpy()
            
        # Department prediction derived from category prediction mapping
        pred_cat_names = fe.decode_labels("category", mt_preds["category"])
        dept_map = {
            "Account/Login": "Technical Support", "Payment": "Billing", "Refund": "Billing",
            "Order": "Logistics", "Delivery": "Logistics", "Product": "Product Support",
            "Technical Support": "Technical Support", "Subscription": "Customer Success",
            "Cancellation": "Customer Success", "Security/Fraud": "Security"
        }
        pred_dept_names = [dept_map.get(c, "Technical Support") for c in pred_cat_names]
        mt_preds["department"] = fe.label_encoders["department"].transform(pred_dept_names)
        
        mt_res = {}
        for target in ["category", "priority", "sentiment", "department"]:
            acc = accuracy_score(y_test_encoded[target], mt_preds[target])
            p, r, f1, _ = precision_recall_fscore_support(y_test_encoded[target], mt_preds[target], average="weighted")
            mt_res[target] = {"accuracy": float(acc), "precision": float(p), "recall": float(r), "f1_score": float(f1)}
        
        results_summary["multi_task_bilstm"] = mt_res
        
        # Plot Confusion Matrices for Multi-Task BiLSTM
        for target in ["category", "priority", "sentiment", "department"]:
            labels = list(fe.label_encoders[target].classes_)
            save_p = os.path.join(FIGURES_DIR, f"{target}_confusion_matrix.png")
            plot_confusion_matrix(y_test_encoded[target], mt_preds[target], labels, f"Confusion Matrix: {target.capitalize()}", save_p)

    # Plot Model Comparison Bar Charts
    plot_model_comparison(results_summary)
    plot_dl_curves()
    
    # Save metrics JSON
    with open("reports/metrics_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    print("Saved evaluation metrics summary -> reports/metrics_summary.json")

if __name__ == "__main__":
    run_comprehensive_evaluation()
