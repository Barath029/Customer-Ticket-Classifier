import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import json
import torch
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

from src.preprocessing import SimpleTokenizer, clean_text
from src.nlp_features import FeatureExtractor
from src.ml_models import MultiTargetMLClassifier
from src.dl_models import MultiTaskBiLSTM
from src.response_generator import generate_suggested_response

# Page configuration
st.set_page_config(
    page_title="AI Ticket Classifier | Support Triage System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Mode CSS Styling with Glassmorphism
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Banner */
    .header-box {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(67, 56, 202, 0.3);
        margin-bottom: 25px;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    .header-title {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        color: #c7d2fe;
        font-size: 1.05rem;
        margin-top: 6px;
    }
    
    /* Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(99, 102, 241, 0.5);
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .metric-val {
        color: #f8fafc;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 6px;
    }
    
    /* Priority Badges */
    .badge-urgent { background-color: #ef4444; color: #fff; padding: 4px 10px; border-radius: 6px; font-weight: 700; }
    .badge-high { background-color: #f97316; color: #fff; padding: 4px 10px; border-radius: 6px; font-weight: 700; }
    .badge-medium { background-color: #eab308; color: #000; padding: 4px 10px; border-radius: 6px; font-weight: 700; }
    .badge-low { background-color: #10b981; color: #fff; padding: 4px 10px; border-radius: 6px; font-weight: 700; }
    
    /* Sentiment Badges */
    .sentiment-positive { color: #34d399; font-weight: 700; }
    .sentiment-neutral { color: #94a3b8; font-weight: 700; }
    .sentiment-negative { color: #f87171; font-weight: 700; }
    
    /* Response Container */
    .response-box {
        background: #1e293b;
        border-left: 5px solid #6366f1;
        border-radius: 10px;
        padding: 20px;
        font-size: 0.98rem;
        line-height: 1.6;
        color: #e2e8f0;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Load Trained Models & Artifacts with Caching
@st.cache_resource
def load_all_artifacts():
    fe = FeatureExtractor.load("models/feature_extractor.pkl") if os.path.exists("models/feature_extractor.pkl") else None
    
    # ML Models
    ml_models = {}
    for algo in ["logistic_regression", "naive_bayes", "svm"]:
        p = os.path.join("models/ml", f"{algo}_model.pkl")
        if os.path.exists(p):
            ml_models[algo] = MultiTargetMLClassifier.load(p)
            
    # DL Multi-Task Model
    dl_model = None
    tokenizer = None
    if os.path.exists("models/dl/multitask_bilstm.pt") and os.path.exists("models/dl/tokenizer.pkl"):
        tokenizer = joblib.load("models/dl/tokenizer.pkl")
        dl_model = MultiTaskBiLSTM(
            vocab_size=len(tokenizer.word2idx),
            num_categories=len(fe.label_encoders["category"].classes_),
            num_priorities=len(fe.label_encoders["priority"].classes_),
            num_sentiments=len(fe.label_encoders["sentiment"].classes_)
        )
        dl_model.load_state_dict(torch.load("models/dl/multitask_bilstm.pt", map_location=torch.device('cpu')))
        dl_model.eval()
        
    return fe, ml_models, dl_model, tokenizer

fe, ml_models, dl_model, tokenizer = load_all_artifacts()

# Header Banner
st.markdown("""
<div class="header-box">
    <div class="header-title">⚡ AI Customer Support Ticket Classifier</div>
    <div class="header-subtitle">Automated NLP & Deep Learning Triage, Department Routing, and AI Response Generator</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select Module:", [
    "🎫 Live Ticket Classifier",
    "📁 Batch Ticket Processing",
    "📊 Model Comparisons & Metrics",
    "🔍 NLP & Dataset Analytics"
])

model_choice = st.sidebar.selectbox(
    "Classifier Engine:",
    ["Multi-Task BiLSTM (Deep Learning)", "Logistic Regression (ML Baseline)", "Naive Bayes (ML Baseline)", "SVM (ML Baseline)"]
)

# Sidebar System Specs
st.sidebar.markdown("---")
st.sidebar.markdown("**System Details**")
st.sidebar.caption("• Architecture: BiLSTM + Shared Dense")
st.sidebar.caption("• Framework: PyTorch & Scikit-Learn")
st.sidebar.caption("• Target Labels: Category, Priority, Sentiment, Department")

# Department mapping dictionary
DEPT_MAP = {
    "Account/Login": "Technical Support", "Payment": "Billing", "Refund": "Billing",
    "Order": "Logistics", "Delivery": "Logistics", "Product": "Product Support",
    "Technical Support": "Technical Support", "Subscription": "Customer Success",
    "Cancellation": "Customer Success", "Security/Fraud": "Security"
}


# ==========================================
# MODULE 1: LIVE TICKET CLASSIFIER
# ==========================================
if app_mode == "🎫 Live Ticket Classifier":
    st.subheader("Interactive Ticket Triage")
    st.markdown("Enter a customer support query below or choose a preset example to analyze predictions.")
    
    # Preset Examples
    col_ex1, col_ex2, col_ex3, col_ex4 = st.columns(4)
    preset_text = ""
    if col_ex1.button("💳 Double Payment"):
        preset_text = "I was charged twice for the same order and I still haven't received my refund."
    if col_ex2.button("🔑 Account Locked"):
        preset_text = "I forgot my password and cannot access my account despite multiple reset attempts."
    if col_ex3.button("📦 Package Delayed"):
        preset_text = "The package was supposed to arrive yesterday but shipment tracking hasn't updated."
    if col_ex4.button("❌ Cancel Plan"):
        preset_text = "I want to cancel my premium subscription immediately before next billing cycle."

    input_ticket = st.text_area(
        "Customer Support Ticket Text:",
        value=preset_text if preset_text else "I was charged twice for order #ORD-88219 and the payment failed on website.",
        height=120,
        placeholder="Type customer complaint here..."
    )
    
    if st.button("🚀 Analyze Ticket", type="primary"):
        if not input_ticket.strip():
            st.warning("Please enter ticket text before submitting.")
        else:
            with st.spinner("Processing text through NLP pipeline & Deep Learning Neural Network..."):
                pred_category, pred_priority, pred_sentiment, pred_dept = "", "", "", ""
                confidence_score = 95.0
                
                # Model Inference
                if "BiLSTM" in model_choice and dl_model is not None and tokenizer is not None:
                    seq = tokenizer.texts_to_sequences([input_ticket], max_len=50)
                    t_seq = torch.tensor(seq, dtype=torch.long)
                    with torch.no_grad():
                        out = dl_model(t_seq)
                        
                    c_idx = torch.argmax(out["category"], dim=1).item()
                    p_idx = torch.argmax(out["priority"], dim=1).item()
                    s_idx = torch.argmax(out["sentiment"], dim=1).item()
                    
                    pred_category = fe.label_encoders["category"].inverse_transform([c_idx])[0]
                    pred_priority = fe.label_encoders["priority"].inverse_transform([p_idx])[0]
                    pred_sentiment = fe.label_encoders["sentiment"].inverse_transform([s_idx])[0]
                    pred_dept = DEPT_MAP.get(pred_category, "Technical Support")
                    
                    # Compute confidence
                    probs = torch.softmax(out["category"], dim=1)[0]
                    confidence_score = float(probs[c_idx].item() * 100)
                else:
                    # ML Model Fallback
                    algo_key = "logistic_regression"
                    if "Naive Bayes" in model_choice: algo_key = "naive_bayes"
                    elif "SVM" in model_choice: algo_key = "svm"
                    
                    if algo_key in ml_models:
                        clf = ml_models[algo_key]
                        X_tfidf = fe.transform_tfidf([input_ticket])
                        preds = clf.predict(X_tfidf)
                        
                        pred_category = fe.label_encoders["category"].inverse_transform([preds["category"][0]])[0]
                        pred_priority = fe.label_encoders["priority"].inverse_transform([preds["priority"][0]])[0]
                        pred_sentiment = fe.label_encoders["sentiment"].inverse_transform([preds["sentiment"][0]])[0]
                        pred_dept = DEPT_MAP.get(pred_category, "Technical Support")
                        
                        probs = clf.predict_proba(X_tfidf)["category"][0]
                        confidence_score = float(np.max(probs) * 100)

            # Display Prediction Cards
            st.markdown("### Prediction Results")
            c1, c2, c3, c4, c5 = st.columns(5)
            
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Category</div>
                    <div class="metric-val" style="color: #818cf8;">{pred_category}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with c2:
                badge_cls = f"badge-{pred_priority.lower()}"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Priority</div>
                    <div class="metric-val"><span class="{badge_cls}">{pred_priority.upper()}</span></div>
                </div>
                """, unsafe_allow_html=True)
                
            with c3:
                sen_cls = f"sentiment-{pred_sentiment.lower()}"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Sentiment</div>
                    <div class="metric-val"><span class="{sen_cls}">{pred_sentiment}</span></div>
                </div>
                """, unsafe_allow_html=True)
                
            with c4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Department</div>
                    <div class="metric-val" style="color: #38bdf8;">{pred_dept}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with c5:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-val" style="color: #34d399;">{confidence_score:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(min(confidence_score / 100.0, 1.0))

            # Suggested Response Box
            st.markdown("### 🤖 Suggested AI Support Response")
            suggested_resp = generate_suggested_response(pred_category, pred_priority, pred_sentiment, pred_dept, input_ticket)
            formatted_resp = suggested_resp.replace("\n", "<br>")
            st.markdown(f'<div class="response-box">{formatted_resp}</div>', unsafe_allow_html=True)


# ==========================================
# MODULE 2: BATCH TICKET PROCESSING
# ==========================================
elif app_mode == "📁 Batch Ticket Processing":
    st.subheader("Batch Ticket Classification")
    st.markdown("Upload a CSV file containing a `ticket_text` column to perform bulk multi-task classification.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            # Clean column headers (strip whitespace and BOM)
            df_batch.columns = df_batch.columns.astype(str).str.strip().str.replace('\ufeff', '')
            
            # Smart detection of ticket text column
            text_col = None
            for col in df_batch.columns:
                norm = col.lower().replace(" ", "_").replace("-", "_")
                if norm in ["ticket_text", "tickettext", "text", "ticket", "complaint", "message", "query", "description"]:
                    text_col = col
                    break
                    
            if text_col is None:
                obj_cols = df_batch.select_dtypes(include=['object', 'string']).columns
                if len(obj_cols) > 0:
                    text_col = obj_cols[0]

            if text_col is None:
                st.error("Uploaded CSV must contain a text/ticket column (e.g., `ticket_text`).")
            else:
                st.success(f"Loaded {len(df_batch)} tickets for batch analysis using column: `{text_col}`.")
                if st.button("⚡ Run Batch Classification"):
                    with st.spinner("Classifying tickets..."):
                        cat_preds, pri_preds, sen_preds, dept_preds = [], [], [], []
                        
                        for text in df_batch[text_col]:
                            seq = tokenizer.texts_to_sequences([str(text)], max_len=50)
                            t_seq = torch.tensor(seq, dtype=torch.long)
                            with torch.no_grad():
                                out = dl_model(t_seq)
                            
                            c_idx = torch.argmax(out["category"], dim=1).item()
                            p_idx = torch.argmax(out["priority"], dim=1).item()
                            s_idx = torch.argmax(out["sentiment"], dim=1).item()
                            
                            cat = fe.label_encoders["category"].inverse_transform([c_idx])[0]
                            pri = fe.label_encoders["priority"].inverse_transform([p_idx])[0]
                            sen = fe.label_encoders["sentiment"].inverse_transform([s_idx])[0]
                            dept = DEPT_MAP.get(cat, "Technical Support")
                            
                            cat_preds.append(cat)
                            pri_preds.append(pri)
                            sen_preds.append(sen)
                            dept_preds.append(dept)
                            
                        df_batch["Predicted_Category"] = cat_preds
                        df_batch["Predicted_Priority"] = pri_preds
                        df_batch["Predicted_Sentiment"] = sen_preds
                        df_batch["Predicted_Department"] = dept_preds
                        
                        st.dataframe(df_batch, use_container_width=True)
                        
                        csv_data = df_batch.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Classified CSV",
                            data=csv_data,
                            file_name="classified_tickets.csv",
                            mime="text/csv"
                        )
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")


# ==========================================
# MODULE 3: MODEL COMPARISONS & METRICS
# ==========================================
elif app_mode == "📊 Model Comparisons & Metrics":
    st.subheader("Performance Evaluation & Architecture Comparisons")
    
    metrics_path = "reports/metrics_summary.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)
            
        st.markdown("### Test Set Accuracy & Weighted F1 Summary")
        rows = []
        for model_name, target_dict in metrics_data.items():
            rows.append({
                "Model Architecture": model_name.upper().replace("_", " "),
                "Category Acc": f"{target_dict['category']['accuracy']*100:.2f}%",
                "Category F1": f"{target_dict['category']['f1_score']:.4f}",
                "Priority Acc": f"{target_dict['priority']['accuracy']*100:.2f}%",
                "Sentiment Acc": f"{target_dict['sentiment']['accuracy']*100:.2f}%",
                "Department Acc": f"{target_dict['department']['accuracy']*100:.2f}%"
            })
        st.table(pd.DataFrame(rows))
        
    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists("reports/figures/model_comparison_accuracy.png"):
            st.image("reports/figures/model_comparison_accuracy.png", caption="Category Classification Accuracy", use_container_width=True)
    with c2:
        if os.path.exists("reports/figures/model_comparison_f1.png"):
            st.image("reports/figures/model_comparison_f1.png", caption="Category Classification F1 Score", use_container_width=True)

    st.markdown("---")
    st.markdown("### Training Curves & Confusion Matrices")
    c3, c4 = st.columns(2)
    with c3:
        if os.path.exists("reports/figures/training_loss_curves.png"):
            st.image("reports/figures/training_loss_curves.png", caption="Deep Learning Loss Curves", use_container_width=True)
    with c4:
        if os.path.exists("reports/figures/category_confusion_matrix.png"):
            st.image("reports/figures/category_confusion_matrix.png", caption="Category Confusion Matrix", use_container_width=True)


# ==========================================
# MODULE 4: NLP & DATASET ANALYTICS
# ==========================================
elif app_mode == "🔍 NLP & Dataset Analytics":
    st.subheader("NLP Exploratory Data Analysis")
    
    col_a, col_b = st.columns(2)
    with col_a:
        if os.path.exists("reports/figures/wordcloud_tickets.png"):
            st.image("reports/figures/wordcloud_tickets.png", caption="Word Cloud of Ticket Corpus", use_container_width=True)
    with col_b:
        if os.path.exists("reports/figures/top_tfidf_ngrams.png"):
            st.image("reports/figures/top_tfidf_ngrams.png", caption="Top 15 TF-IDF N-Grams", use_container_width=True)
            
    if os.path.exists("data/customer_tickets.csv"):
        st.markdown("### Dataset Sample Explorer")
        df_sample = pd.read_csv("data/customer_tickets.csv")
        st.dataframe(df_sample.head(20), use_container_width=True)
