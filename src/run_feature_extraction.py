"""
run_feature_extraction.py
-------------------------
Fits BoW and TF-IDF on training data, transforms all splits,
and saves everything for use by the classifier training script.

BERT embeddings are generated on Colab (GPU) — there's a note at the end.
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_PROCESSED_DIR, MODELS_DIR
from src.feature_extractor import (
    build_bow_extractor,
    build_tfidf_extractor,
    save_extractor,
    save_features
)


def load_split(split_name: str):
    """Load a cleaned split and return texts + labels."""
    path = os.path.join(DATA_PROCESSED_DIR, f"{split_name}_clean.csv")
    df   = pd.read_csv(path)

    # Drop the 2-3 empty rows from cleaning
    df   = df[df['clean_text'].notna()]
    df   = df[df['clean_text'].str.strip() != '']

    texts  = df['clean_text'].tolist()
    labels = df['label'].tolist()
    print(f"Loaded {split_name}: {len(texts)} samples")
    return texts, labels


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Load all splits ────────────────────────────────────────
    print("Loading data...")
    train_texts, train_labels = load_split('train')
    val_texts,   val_labels   = load_split('val')
    test_texts,  test_labels  = load_split('test')

    # Save labels as numpy arrays for use in training
    np.save(os.path.join(MODELS_DIR, 'train_labels.npy'), np.array(train_labels))
    np.save(os.path.join(MODELS_DIR, 'val_labels.npy'),   np.array(val_labels))
    np.save(os.path.join(MODELS_DIR, 'test_labels.npy'),  np.array(test_labels))
    print("Labels saved.\n")

    # ══════════════════════════════════════════════════════════
    # 1. BAG OF WORDS
    # ══════════════════════════════════════════════════════════
    print("=" * 50)
    print("EXTRACTING: Bag of Words")
    print("=" * 50)

    bow = build_bow_extractor(max_features=5000)

    # IMPORTANT: fit ONLY on training data
    # then transform all splits using the same vocabulary
    X_train_bow = bow.fit_transform(train_texts)
    X_val_bow   = bow.transform(val_texts)
    X_test_bow  = bow.transform(test_texts)

    print(f"BoW feature shape  (train) : {X_train_bow.shape}")
    print(f"BoW feature shape  (val)   : {X_val_bow.shape}")

    # Save extractor and features
    save_extractor(bow, 'bow_extractor.pkl')

    # BoW produces sparse matrices — save differently
    from scipy.sparse import save_npz
    save_npz(os.path.join(MODELS_DIR, 'X_train_bow.npz'), X_train_bow)
    save_npz(os.path.join(MODELS_DIR, 'X_val_bow.npz'),   X_val_bow)
    save_npz(os.path.join(MODELS_DIR, 'X_test_bow.npz'),  X_test_bow)
    print("BoW features saved.\n")

    # ══════════════════════════════════════════════════════════
    # 2. TF-IDF  (three scales: 512, 1024, 2048)
    # ══════════════════════════════════════════════════════════
    print("=" * 50)
    print("EXTRACTING: TF-IDF")
    print("=" * 50)

    for n_components in [512, 1024, 2048]:
        print(f"\n--- TF-IDF with SVD({n_components}) ---")

        tfidf = build_tfidf_extractor(n_components=n_components)

        X_train_tfidf = tfidf.fit_transform(train_texts)
        X_val_tfidf   = tfidf.transform(val_texts)
        X_test_tfidf  = tfidf.transform(test_texts)

        print(f"TF-IDF shape (train): {X_train_tfidf.shape}")

        save_extractor(tfidf, f'tfidf_{n_components}_extractor.pkl')
        save_features(X_train_tfidf, f'X_train_tfidf_{n_components}.npy')
        save_features(X_val_tfidf,   f'X_val_tfidf_{n_components}.npy')
        save_features(X_test_tfidf,  f'X_test_tfidf_{n_components}.npy')

    print("\nAll TF-IDF features saved.")

    # ══════════════════════════════════════════════════════════
    # BERT — reminder
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 50)
    print("NOTE: BERT embeddings")
    print("=" * 50)
    print("BERT embedding extraction is slow on CPU (~2-3 hours for 17k samples).")
    print("We will generate these on Google Colab (free GPU) in Phase 5.")
    print("For now, BoW and TF-IDF features are ready for traditional ML models.")


if __name__ == "__main__":
    main()