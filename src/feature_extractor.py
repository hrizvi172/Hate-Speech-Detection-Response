"""
feature_extractor.py
--------------------
Three feature extraction methods:
  1. Bag of Words  (BoW)
  2. TF-IDF        (multiple scales)
  3. BERT embeddings  (contextual, most powerful)

Usage:
  from src.feature_extractor import FeatureExtractor
  fe = FeatureExtractor(method='tfidf')
  X_train = fe.fit_transform(train_texts)
  X_test  = fe.transform(test_texts)
"""

import os
import sys
import pickle
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODELS_DIR, BERT_MODEL_NAME, MAX_TOKEN_LENGTH


# ══════════════════════════════════════════════════════════════════════════════
# 1. BAG OF WORDS
# ══════════════════════════════════════════════════════════════════════════════

def build_bow_extractor(max_features: int = 5000):
    """
    CountVectorizer with unigrams + bigrams.
    max_features = vocabulary size (keep top 5000 most common words/phrases)
    ngram_range=(1,2) means it captures:
      - single words   : "hate", "speech"
      - two-word pairs : "hate speech", "violent content"
    """
    return CountVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),       # unigrams and bigrams
        min_df=2,                 # ignore terms that appear in < 2 documents
        max_df=0.95               # ignore terms that appear in > 95% of docs
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. TF-IDF  (multiple scales as per proposal)
# ══════════════════════════════════════════════════════════════════════════════

def build_tfidf_extractor(n_components: int = 512):
    """
    TF-IDF vectorizer followed by SVD dimensionality reduction.

    Why SVD?
    --------
    Raw TF-IDF can produce 50,000+ dimensions (one per word).
    SVD compresses this to n_components (512/1024/2048) while
    keeping the most important information — like JPEG compression for text.

    sublinear_tf=True : uses 1 + log(tf) instead of raw tf
                        prevents very frequent words from dominating
    """
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95
    )
    svd = TruncatedSVD(n_components=n_components, random_state=42)

    # Pipeline chains them: vectorize → reduce dimensions
    return Pipeline([
        ('tfidf', vectorizer),
        ('svd',   svd)
    ])


# ══════════════════════════════════════════════════════════════════════════════
# 3. BERT EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════

def get_bert_embeddings(texts: list,
                        model_name: str = BERT_MODEL_NAME,
                        max_length: int = MAX_TOKEN_LENGTH,
                        batch_size: int = 32) -> np.ndarray:
    """
    Converts a list of texts into BERT embeddings.

    How it works:
    -------------
    BERT reads the full sentence and produces a 768-dimensional vector
    that captures the meaning of the whole sentence (the [CLS] token).
    Unlike BoW/TF-IDF, it understands context and word order.

    We process in batches to avoid running out of memory.

    Returns: numpy array of shape (n_samples, 768)
    """
    # Import here so the file can be imported even without torch installed
    import torch
    from transformers import AutoTokenizer, AutoModel

    print(f"Loading BERT model: {model_name}")
    print("(First time will download ~440MB — this is normal)\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name)

    # Use GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()   # evaluation mode — no gradient tracking needed

    all_embeddings = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]

        # Show progress every 10 batches
        if i % (batch_size * 10) == 0:
            print(f"  Processing samples {i} – {min(i+batch_size, len(texts))} "
                  f"/ {len(texts)}")

        # Tokenize: convert text → token IDs + attention masks
        encoded = tokenizer(
            batch,
            padding=True,        # pad shorter sequences
            truncation=True,     # cut sequences longer than max_length
            max_length=max_length,
            return_tensors='pt'  # return PyTorch tensors
        )

        # Move to same device as model
        input_ids      = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        # Forward pass — no gradient calculation needed for inference
        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                           attention_mask=attention_mask)

        # outputs.last_hidden_state shape: (batch, seq_len, 768)
        # We take the [CLS] token (index 0) as the sentence representation
        cls_embeddings = outputs.last_hidden_state[:, 0, :]

        # Move back to CPU and convert to numpy
        all_embeddings.append(cls_embeddings.cpu().numpy())

    embeddings = np.vstack(all_embeddings)
    print(f"Done. Embedding shape: {embeddings.shape}")
    return embeddings


# ══════════════════════════════════════════════════════════════════════════════
# SAVE / LOAD HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def save_features(data: np.ndarray, filename: str) -> None:
    """Save a numpy array to the outputs folder."""
    path = os.path.join(MODELS_DIR, filename)
    np.save(path, data)
    print(f"Saved features → {path}")


def load_features(filename: str) -> np.ndarray:
    """Load a numpy array from the outputs folder."""
    path = os.path.join(MODELS_DIR, filename)
    return np.load(path)


def save_extractor(extractor, filename: str) -> None:
    """Save a fitted BoW/TF-IDF extractor using pickle."""
    path = os.path.join(MODELS_DIR, 'classifiers', filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(extractor, f)
    print(f"Saved extractor → {path}")


def load_extractor(filename: str):
    """Load a fitted extractor."""
    path = os.path.join(MODELS_DIR, 'classifiers', filename)
    with open(path, 'rb') as f:
        return pickle.load(f)