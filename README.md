# ⚡ AI Customer Support Ticket Classifier

An intelligent, multi-task **NLP & Deep Learning Customer Support Ticket Classifier** that automatically analyzes incoming customer tickets and predicts:
1. **Category** (Account/Login, Payment, Refund, Order, Delivery, Product, Technical Support, Subscription, Cancellation, Security/Fraud)
2. **Priority** (Low, Medium, High, Urgent)
3. **Sentiment** (Positive, Neutral, Negative)
4. **Department** (Technical Support, Billing, Logistics, Product Support, Customer Success, Security)

Includes baseline Machine Learning classifiers (Naive Bayes, Logistic Regression, Linear SVM), single-task Deep Learning models (RNN, LSTM, GRU, BiLSTM), a **Multi-Task BiLSTM Neural Network**, an **AI Suggested Response Generator**, a rich **Streamlit Web Dashboard**, and **Two Comprehensive Academic Reports**.

---

## 📁 Repository Structure

```text
DL & NLP MiniProject/
├── data/
│   ├── dataset_generator.py      # Programmatic generator for 4,200 synthetic tickets
│   ├── customer_tickets.csv      # Complete synthetic dataset
│   ├── train.csv                 # 70% Training set (2,940 rows)
│   ├── val.csv                   # 15% Validation set (630 rows)
│   └── test.csv                  # 15% Test set (630 rows)
├── src/
│   ├── preprocessing.py          # NLTK cleaning, lemmatization, tokenization
│   ├── nlp_features.py           # TF-IDF vectorization & Label Encoding
│   ├── ml_models.py              # Baseline ML models (Naive Bayes, Logistic Regression, Linear SVM)
│   ├── dl_models.py              # PyTorch RNN, LSTM, GRU, BiLSTM, Multi-Task BiLSTM architectures
│   ├── train_dl.py               # Deep learning training pipeline & loss logging
│   ├── evaluate.py               # Evaluation engine, metrics calculation, figure generation
│   └── response_generator.py     # AI template response recommendation engine
├── models/                       # Saved trained model artifacts (.pkl, .pt)
├── reports/
│   ├── figures/                  # Confusion matrices, loss curves, bar charts, word cloud
│   ├── NLP_REPORT.md             # Full academic report focusing on NLP methodology
│   ├── DEEP_LEARNING_REPORT.md   # Full academic report focusing on Deep Learning methodology
│   └── metrics_summary.json      # Consolidated numerical metrics across all models
├── app.py                        # Interactive Streamlit Web Application
├── requirements.txt              # Dependency specifications
└── README.md                     # Project documentation & execution guide
```

---

## 🚀 Quick Start Guide

### 1. Installation & Environment Setup
Ensure Python 3.10+ is installed, then install required packages:

```bash
pip install -r requirements.txt
```

### 2. Generate Dataset Splits
Generate 4,200 synthetic tickets and create `train.csv`, `val.csv`, `test.csv`:

```bash
python data/dataset_generator.py
```

### 3. Train Baseline Machine Learning Models
Train Naive Bayes, Logistic Regression, and Linear SVM models:

```bash
python src/ml_models.py
```

### 4. Train Deep Learning Models
Train single-task RNN, LSTM, GRU, BiLSTM, and the Multi-Task BiLSTM neural network:

```bash
python src/train_dl.py
```

### 5. Run Comprehensive Model Evaluation & Plot Figures
Evaluate models on the test set and generate all charts in `reports/figures/`:

```bash
python src/evaluate.py
```

### 6. Launch Interactive Streamlit Web Dashboard
Run the web application dashboard using the Python module launcher:

```bash
python -m streamlit run app.py
```

---

## 📊 Key Results Summary

| Model Architecture | Category Acc | Category F1 | Priority Acc | Sentiment Acc | Department Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Bayes (TF-IDF)** | 100.00% | 1.0000 | 94.92% | 93.49% | 100.00% |
| **Logistic Regression (TF-IDF)** | 100.00% | 1.0000 | 94.92% | 93.49% | 100.00% |
| **Linear SVM (TF-IDF)** | 100.00% | 1.0000 | 94.92% | 93.49% | 100.00% |
| **Multi-Task BiLSTM (PyTorch)** | **99.84%** | **0.9984** | **95.24%** | **94.12%** | **99.84%** |

---

## 📘 Academic Reports
- **NLP Report:** `reports/NLP_REPORT.md`
- **Deep Learning Report:** `reports/DEEP_LEARNING_REPORT.md`
