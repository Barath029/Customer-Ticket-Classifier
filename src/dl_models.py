import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleTaskRNN(nn.Module):
    """Single-task RNN model (supports rnn_type: 'rnn', 'lstm', 'gru', 'bilstm')."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, rnn_type="bilstm", dropout=0.3):
        super(SingleTaskRNN, self).__init__()
        self.rnn_type = rnn_type.lower()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        is_bidirectional = "bilstm" in self.rnn_type
        num_directions = 2 if is_bidirectional else 1
        
        if "gru" in self.rnn_type:
            self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=is_bidirectional)
        elif "lstm" in self.rnn_type or "bilstm" in self.rnn_type:
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=is_bidirectional)
        else:
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, bidirectional=is_bidirectional)
            
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * num_directions, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(x)) # [batch_size, seq_len, embed_dim]
        
        if isinstance(self.rnn, nn.LSTM):
            out, (hn, cn) = self.rnn(embedded)
        else:
            out, hn = self.rnn(embedded)
            
        # Global max pooling across sequence dimension for rich representations
        pooled, _ = torch.max(out, dim=1) # [batch_size, hidden_dim * num_directions]
        logits = self.fc(self.dropout(pooled))
        return logits


class MultiTaskBiLSTM(nn.Module):
    """
    Multi-Task Deep Learning Model:
    Shared Embedding + Shared BiLSTM -> Shared Latent Representation -> 3 Softmax Heads
    1. Category Head (num_categories)
    2. Priority Head (num_priorities)
    3. Sentiment Head (num_sentiments)
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, 
                 num_categories=10, num_priorities=4, num_sentiments=3, dropout=0.3):
        super(MultiTaskBiLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            batch_first=True, 
            bidirectional=True,
            num_layers=2,
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Shared dense representation layer
        shared_dim = hidden_dim * 2
        self.shared_dense = nn.Linear(shared_dim, 128)
        self.relu = nn.ReLU()
        
        # Classification Heads
        self.category_head = nn.Linear(128, num_categories)
        self.priority_head = nn.Linear(128, num_priorities)
        self.sentiment_head = nn.Linear(128, num_sentiments)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x) # [batch_size, seq_len, embed_dim]
        lstm_out, _ = self.bilstm(embedded) # [batch_size, seq_len, hidden_dim * 2]
        
        # Max pooling across sequence
        pooled, _ = torch.max(lstm_out, dim=1) # [batch_size, hidden_dim * 2]
        
        shared_feat = self.relu(self.shared_dense(self.dropout(pooled)))
        shared_feat = self.dropout(shared_feat)
        
        cat_logits = self.category_head(shared_feat)
        pri_logits = self.priority_head(shared_feat)
        sen_logits = self.sentiment_head(shared_feat)
        
        return {
            "category": cat_logits,
            "priority": pri_logits,
            "sentiment": sen_logits
        }

if __name__ == "__main__":
    vocab_size = 1000
    model = MultiTaskBiLSTM(vocab_size=vocab_size)
    dummy_input = torch.randint(0, vocab_size, (8, 50))
    outputs = model(dummy_input)
    print("MultiTaskBiLSTM Outputs:")
    for k, v in outputs.items():
        print(f"  {k:10s} logits shape: {v.shape}")
