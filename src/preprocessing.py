import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK resources automatically if missing
for resource in ['punkt', 'stopwords', 'wordnet', 'punkt_tab']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

try:
    STOP_WORDS = set(stopwords.words('english'))
except Exception:
    STOP_WORDS = {"i", "me", "my", "myself", "we", "our", "ours", "you", "your", "yours", 
                  "he", "him", "his", "she", "her", "it", "its", "they", "them", "their", 
                  "what", "which", "who", "whom", "this", "that", "these", "those", "am", 
                  "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", 
                  "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", 
                  "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", 
                  "against", "between", "into", "through", "during", "before", "after", 
                  "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", 
                  "over", "under", "again", "further", "then", "once"}

LEMMATIZER = WordNetLemmatizer()

def clean_text(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
    """Preprocess and clean raw ticket text."""
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove URLs and Email addresses
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    
    # 3. Keep alphanumeric and basic space
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # 4. Tokenization
    tokens = text.split()
    
    # 5. Stopword removal
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
        
    # 6. Lemmatization
    if lemmatize:
        try:
            tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
        except Exception:
            pass
            
    cleaned = " ".join(tokens)
    return cleaned

class SimpleTokenizer:
    """Custom Tokenizer for mapping words to integer indices for PyTorch DL models."""
    def __init__(self, max_vocab_size=10000, unk_token="<UNK>", pad_token="<PAD>"):
        self.max_vocab_size = max_vocab_size
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.word2idx = {pad_token: 0, unk_token: 1}
        self.idx2word = {0: pad_token, 1: unk_token}
        self.word_counts = {}

    def fit_on_texts(self, texts):
        for text in texts:
            tokens = clean_text(text).split()
            for token in tokens:
                self.word_counts[token] = self.word_counts.get(token, 0) + 1

        # Sort by frequency
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        for word, _ in sorted_words[:self.max_vocab_size - 2]:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def texts_to_sequences(self, texts, max_len=50):
        sequences = []
        for text in texts:
            tokens = clean_text(text).split()
            seq = [self.word2idx.get(t, self.word2idx[self.unk_token]) for t in tokens]
            
            # Padding / Truncating
            if len(seq) < max_len:
                seq = seq + [self.word2idx[self.pad_token]] * (max_len - len(seq))
            else:
                seq = seq[:max_len]
            sequences.append(seq)
        return sequences

if __name__ == "__main__":
    sample = "I was charged twice for order #ORD-12345! Please fix this immediately."
    print("Original:", sample)
    print("Cleaned :", clean_text(sample))
