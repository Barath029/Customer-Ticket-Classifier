import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import joblib

from src.preprocessing import SimpleTokenizer
from src.nlp_features import FeatureExtractor, load_data_splits
from src.dl_models import SingleTaskRNN, MultiTaskBiLSTM

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

class TicketDataset(Dataset):
    def __init__(self, sequences, labels_dict=None):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels_dict = labels_dict
        if labels_dict is not None:
            self.category = torch.tensor(labels_dict["category"], dtype=torch.long)
            self.priority = torch.tensor(labels_dict["priority"], dtype=torch.long)
            self.sentiment = torch.tensor(labels_dict["sentiment"], dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = {"sequence": self.sequences[idx]}
        if self.labels_dict is not None:
            item["category"] = self.category[idx]
            item["priority"] = self.priority[idx]
            item["sentiment"] = self.sentiment[idx]
        return item


def train_single_task_model(model, train_loader, val_loader, epochs=10, lr=0.001, device="cpu", target="category"):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            x = batch["sequence"].to(device)
            y = batch[target].to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["sequence"].to(device)
                y = batch[target].to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(y.cpu().numpy())
                
        val_loss /= len(val_loader)
        val_acc = accuracy_score(all_targets, all_preds)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
    return history, val_acc


def train_multitask_bilstm(train_loader, val_loader, tokenizer, fe, epochs=12, lr=0.001, device="cpu"):
    num_cats = len(fe.label_encoders["category"].classes_)
    num_pris = len(fe.label_encoders["priority"].classes_)
    num_sens = len(fe.label_encoders["sentiment"].classes_)
    
    model = MultiTaskBiLSTM(
        vocab_size=len(tokenizer.word2idx),
        embed_dim=128,
        hidden_dim=128,
        num_categories=num_cats,
        num_priorities=num_pris,
        num_sentiments=num_sens,
        dropout=0.3
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {
        "train_loss": [], "val_loss": [],
        "val_category_acc": [], "val_priority_acc": [], "val_sentiment_acc": []
    }
    
    best_combined_acc = 0.0
    model_save_dir = "models/dl"
    os.makedirs(model_save_dir, exist_ok=True)
    
    print("\n=================== TRAINING MULTI-TASK BiLSTM MODEL ===================")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        
        for batch in train_loader:
            x = batch["sequence"].to(device)
            y_cat = batch["category"].to(device)
            y_pri = batch["priority"].to(device)
            y_sen = batch["sentiment"].to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
            
            loss_cat = criterion(outputs["category"], y_cat)
            loss_pri = criterion(outputs["priority"], y_pri)
            loss_sen = criterion(outputs["sentiment"], y_sen)
            
            total_loss = loss_cat + loss_pri + loss_sen
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            
        train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        preds = {"category": [], "priority": [], "sentiment": []}
        targets = {"category": [], "priority": [], "sentiment": []}
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch["sequence"].to(device)
                outputs = model(x)
                
                loss_c = criterion(outputs["category"], batch["category"].to(device))
                loss_p = criterion(outputs["priority"], batch["priority"].to(device))
                loss_s = criterion(outputs["sentiment"], batch["sentiment"].to(device))
                val_loss += (loss_c + loss_p + loss_s).item()
                
                for task in ["category", "priority", "sentiment"]:
                    p = torch.argmax(outputs[task], dim=1).cpu().numpy()
                    preds[task].extend(p)
                    targets[task].extend(batch[task].numpy())
                    
        val_loss /= len(val_loader)
        
        acc_cat = accuracy_score(targets["category"], preds["category"])
        acc_pri = accuracy_score(targets["priority"], preds["priority"])
        acc_sen = accuracy_score(targets["sentiment"], preds["sentiment"])
        comb_acc = (acc_cat + acc_pri + acc_sen) / 3.0
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_category_acc"].append(acc_cat)
        history["val_priority_acc"].append(acc_pri)
        history["val_sentiment_acc"].append(acc_sen)
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Cat Acc: {acc_cat*100:.2f}% | Pri Acc: {acc_pri*100:.2f}% | Sen Acc: {acc_sen*100:.2f}%")
        
        if comb_acc > best_combined_acc:
            best_combined_acc = comb_acc
            torch.save(model.state_dict(), os.path.join(model_save_dir, "multitask_bilstm.pt"))
            
    print(f"\nBest Multi-Task BiLSTM Combined Validation Accuracy: {best_combined_acc*100:.2f}%")
    return model, history


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_df, val_df, test_df = load_data_splits()
    
    fe = FeatureExtractor.load("models/feature_extractor.pkl") if os.path.exists("models/feature_extractor.pkl") else FeatureExtractor()
    fe.fit_labels(train_df)
    
    tokenizer = SimpleTokenizer(max_vocab_size=8000)
    tokenizer.fit_on_texts(train_df["ticket_text"])
    
    # Save tokenizer
    os.makedirs("models/dl", exist_ok=True)
    joblib.dump(tokenizer, "models/dl/tokenizer.pkl")
    
    train_seqs = tokenizer.texts_to_sequences(train_df["ticket_text"], max_len=50)
    val_seqs = tokenizer.texts_to_sequences(val_df["ticket_text"], max_len=50)
    test_seqs = tokenizer.texts_to_sequences(test_df["ticket_text"], max_len=50)
    
    y_train = fe.encode_labels(train_df)
    y_val = fe.encode_labels(val_df)
    y_test = fe.encode_labels(test_df)
    
    train_ds = TicketDataset(train_seqs, y_train)
    val_ds = TicketDataset(val_seqs, y_val)
    test_ds = TicketDataset(test_seqs, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    # Train Single-Task DL Baselines for Category Classification comparison
    single_models = ["rnn", "lstm", "gru", "bilstm"]
    single_histories = {}
    
    print("\n=================== TRAINING SINGLE-TASK DL BASELINES ===================")
    num_cats = len(fe.label_encoders["category"].classes_)
    vocab_size = len(tokenizer.word2idx)
    
    for rnn_type in single_models:
        print(f"Training Single-Task {rnn_type.upper()}...")
        st_model = SingleTaskRNN(
            vocab_size=vocab_size,
            embed_dim=128,
            hidden_dim=128,
            num_classes=num_cats,
            rnn_type=rnn_type
        ).to(device)
        
        hist, best_acc = train_single_task_model(st_model, train_loader, val_loader, epochs=8, device=device)
        single_histories[rnn_type] = hist
        print(f"  Single-Task {rnn_type.upper()} Category Val Accuracy: {best_acc*100:.2f}%")
        
    # Train Multi-Task BiLSTM
    mt_model, mt_history = train_multitask_bilstm(train_loader, val_loader, tokenizer, fe, epochs=12, device=device)
    
    # Save training history summary
    full_history = {
        "single_task": single_histories,
        "multi_task": mt_history
    }
    with open("models/dl/history.json", "w") as f:
        json.dump(full_history, f, indent=2)
    print("Saved training history to models/dl/history.json")

if __name__ == "__main__":
    main()
