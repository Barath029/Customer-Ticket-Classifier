# Customer Support Ticket Classification Using Deep Learning & Multi-Task BiLSTM Architectures

**Project Title:** AI Customer Support Ticket Classifier — Deep Learning Academic Report  
**Author:** AI & Deep Learning Research Group  
**Date:** Academic Year 2026  

---

## 1. Executive Summary & Introduction to Deep Learning in NLP
Traditional Machine Learning models rely heavily on hand-crafted feature engineering (such as Bag-of-Words and TF-IDF matrices). While effective for basic keyword matching, TF-IDF representations treat text as an unordered bag of tokens, failing to capture long-range contextual dependencies, word order nuances, semantic similarities, and compositional semantics.

**Deep Learning (DL)** overcomes these limitations by learning dense continuous vector spaces (Word Embeddings) and sequential context representations using Recurrent Neural Architectures. This report presents an advanced **Multi-Task Learning Bidirectional LSTM (BiLSTM)** framework designed to simultaneously predict customer support issue **Category, Priority, and Sentiment** from a single shared latent representation.

---

## 2. Problem Statement & Motivation for Deep Learning
In real-world customer support messages, semantic intent is often expressed through complex phrasing, negation, dynamic context, and subtle urgency cues:

> *"I was planning to recommend your service to my team, but after being charged twice with zero response from support for 3 days, I need this canceled now."*

A static bag-of-words model struggles to differentiate between positive recommendations and urgent cancellation demands embedded within the same document. A **Bidirectional LSTM** processes token sequences forward and backward simultaneously, preserving semantic context across long sequences.

---

## 3. Dataset & Tensor Representation
The dataset of 4,200 customer support tickets is preprocessed and tokenized into integer sequence tensors $\mathbf{X} \in \mathbb{Z}^{B \times L}$, where batch size $B=32$ and fixed sequence length $L=50$. Shorter texts are zero-padded (`<PAD>`), while longer texts are truncated.

---

## 4. Word Embedding Layer
Rather than using static one-hot encoding vectors of dimension $|V|=8,000$, a trainable continuous **Embedding Layer** maps discrete integer indices into a dense vector space:

$$\mathbf{E}: \mathbb{Z} \to \mathbb{R}^{d_e}$$

where embedding dimension $d_e = 128$. This dense mapping allows the neural network to capture semantic proximity:

$$\text{cosine\_sim}(\mathbf{E}(\text{"payment"}), \mathbf{E}(\text{"billing"})) \gg \text{cosine\_sim}(\mathbf{E}(\text{"payment"}), \mathbf{E}(\text{"shipping"}))$$

---

## 5. Recurrent Neural Networks (RNN) Baseline
Standard Recurrent Neural Networks process sequential inputs $\mathbf{x}_t$ iteratively by updating hidden state state $\mathbf{h}_t$:

$$\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)$$

However, standard RNNs suffer from **vanishing and exploding gradients** over long sequences, rendering them unable to retain information beyond 10-15 tokens.

---

## 6. Long Short-Term Memory Networks (LSTM)
LSTMs introduce specialized memory cells $\mathbf{c}_t$ governed by three gating mechanisms to control information flow:

$$\begin{aligned}
\mathbf{f}_t &= \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \quad &&\text{(Forget Gate)} \\
\mathbf{i}_t &= \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) \quad &&\text{(Input Gate)} \\
\tilde{\mathbf{c}}_t &= \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c) \quad &&\text{(Candidate Cell Update)} \\
\mathbf{c}_t &= \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \quad &&\text{(Cell State Update)} \\
\mathbf{o}_t &= \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) \quad &&\text{(Output Gate)} \\
\mathbf{h}_t &= \mathbf{o}_t \odot \tanh(\mathbf{c}_t) \quad &&\text{(Hidden State Output)}
\end{aligned}$$

---

## 7. Gated Recurrent Units (GRU)
GRU simplifies the LSTM architecture by combining cell state and hidden state into a single vector $\mathbf{h}_t$, controlled by two gates: Update Gate $\mathbf{z}_t$ and Reset Gate $\mathbf{r}_t$. This reduces parameter count while maintaining competitive accuracy.

---

## 8. Bidirectional LSTM (BiLSTM) Architecture
A **Bidirectional LSTM** processes sequence tokens in both temporal directions using two separate hidden layers:
- Forward LSTM $\overrightarrow{\mathbf{h}}_t$: Reads from $t=1 \to L$
- Backward LSTM $\overleftarrow{\mathbf{h}}_t$: Reads from $t=L \to 1$

The final representation at step $t$ is the concatenation:

$$\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t \,;\, \overleftarrow{\mathbf{h}}_t] \in \mathbb{R}^{2 \cdot d_h}$$

where hidden dimension $d_h = 128$.

---

## 9. Multi-Task Learning Neural Network Design
Instead of deploying three separate single-task networks, we construct a unified **Multi-Task Learning (MTL)** neural network.

```text
                     Input Ticket Tokens [B, 50]
                                 ↓
                     Trainable Embedding [128D]
                                 ↓
                     2-Layer BiLSTM [256D]
                                 ↓
                     Global Max Pooling [256D]
                                 ↓
                     Shared Dense Layer [128D] + Dropout(0.3)
                      /          |          \
                     /           |           \
                    ↓            ↓            ↓
               Category       Priority     Sentiment
                 Head           Head         Head
               [10 Out]        [4 Out]      [3 Out]
```

### Shared Latent Advantage:
By training all three classification heads simultaneously, the shared BiLSTM representation learns generalized linguistic properties (e.g., urgency markers reinforcing negative sentiment and billing category predictions).

---

## 10. Training Methodology, Loss Functions, & Optimizers
The total multi-task loss is computed as the sum of cross-entropy losses across all three prediction heads:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{category}} + \mathcal{L}_{\text{priority}} + \mathcal{L}_{\text{sentiment}}$$

$$\mathcal{L}_{\text{task}} = -\sum_{i=1}^{K} y_i \log(\hat{y}_i)$$

- **Optimizer:** Adam ($\beta_1=0.9, \beta_2=0.999$, Learning Rate $\eta = 0.001$)
- **Batch Size:** 32
- **Epochs:** 12

---

## 11. Hyperparameter Optimization & Regularization
- **Dropout Rate (0.3):** Applied after embedding and shared dense layers to prevent over-fitting.
- **Global Max Pooling:** Extracts dominant activation signals across sequence length.
- **Gradient Clipping:** Prevents exploding gradient instabilities during backpropagation through time (BPTT).

---

## 12. Validation Curves & Training Dynamics
During 12 training epochs, multi-task total loss decreased monotonically from `3.412` to `0.084` on training data, while validation accuracy steadily converged across all three heads:
- **Category Validation Accuracy:** Reached **99.84%** by Epoch 6.
- **Priority Validation Accuracy:** Reached **95.24%** by Epoch 8.
- **Sentiment Validation Accuracy:** Reached **94.12%** by Epoch 8.

---

## 13. Model Evaluation & Confusion Matrix Analysis
Evaluating the final Multi-Task BiLSTM checkpoint on the 630 held-out test tickets yielded exceptional precision, recall, and F1 performance:

### Category Classification Matrix Highlights:
- **Payment & Refund:** 100% precision; distinct billing vocabulary cleanly separated from delivery queries.
- **Account/Login & Technical Support:** 99.2% precision; slight overlap resolved by multi-task shared representations.
- **Security/Fraud:** 100% recall; urgent fraud alerts correctly routed with zero false negatives.

---

## 14. Comparative Benchmarks Across Architectures

| Architecture Type | Category Acc | Category F1 | Multi-Task Capable | Parameters |
| :--- | :---: | :---: | :---: | :---: |
| **Simple RNN** | 91.27% | 0.9104 | No | ~1.1M |
| **Single-Task LSTM** | 98.41% | 0.9839 | No | ~1.3M |
| **Single-Task GRU** | 98.73% | 0.9870 | No | ~1.2M |
| **Single-Task BiLSTM** | 99.52% | 0.9951 | No | ~1.5M |
| **Multi-Task BiLSTM (Proposed)** | **99.84%** | **0.9984** | **YES (3 Heads)** | **~1.6M** |

---

## 15. Limitations & Future Extensions
While the Multi-Task BiLSTM achieves state-of-the-art results on support ticket triage, future work will explore:
1. **Transformer Integration (DistilBERT / RoBERTa):** Fine-tuning pre-trained language models for enhanced multi-lingual support.
2. **Generative RAG Response Synthesis:** Pairing classification predictions with Retrieval-Augmented Generation (RAG) over internal FAQ knowledge bases.
