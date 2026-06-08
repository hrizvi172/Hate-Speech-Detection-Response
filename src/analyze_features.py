"""
analyze_features.py
-------------------
Shows top words per class using Logistic Regression on BoW.
We train a fresh LR here just for interpretability — 
it doesn't need to be the best model, just interpretable.
"""

import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import load_npz

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODELS_DIR, OUTPUTS_DIR
from src.feature_extractor import load_extractor
from src.train_classical import load_labels, get_class_weights
from sklearn.linear_model import LogisticRegression


def main():
    # Load BoW features and labels
    print("Loading BoW features...")
    X_train = load_npz(os.path.join(MODELS_DIR, 'X_train_bow.npz'))
    X_val   = load_npz(os.path.join(MODELS_DIR, 'X_val_bow.npz'))
    y_train, y_val, _ = load_labels()

    bow = load_extractor('bow_extractor.pkl')
    feature_names = bow.get_feature_names_out()

    # Train LR specifically for interpretation
    class_weights = get_class_weights(y_train)
    print("Training Logistic Regression for interpretation...")
    lr = LogisticRegression(
        max_iter=1000,
        class_weight=class_weights,
        random_state=42,
        C=1.0
    )
    lr.fit(X_train, y_train)

    val_score = lr.score(X_val, y_val)
    print(f"Validation accuracy: {val_score:.4f}")

    class_names = ['Hate Speech', 'Offensive Language', 'Neither']
    colors      = ['#e74c3c', '#e67e22', '#2ecc71']

    print("\nTop 20 words most associated with each class:\n")
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    for i, (class_name, color, ax) in enumerate(
            zip(class_names, colors, axes)):

        coefs       = lr.coef_[i]
        top_idx     = np.argsort(coefs)[-20:][::-1]
        top_words   = [feature_names[j] for j in top_idx]
        top_scores  = [coefs[j]         for j in top_idx]

        print(f"[{class_name}]")
        for word, score in zip(top_words, top_scores):
            print(f"  {word:30s}  {score:+.3f}")
        print()

        ax.barh(range(20), top_scores[::-1], color=color, edgecolor='black')
        ax.set_yticks(range(20))
        ax.set_yticklabels(top_words[::-1], fontsize=9)
        ax.set_title(f'Top words: {class_name}', fontsize=12)
        ax.set_xlabel('Coefficient weight')

    plt.suptitle(
        'Most Important Words per Class — Logistic Regression + BoW',
        fontsize=14
    )
    plt.tight_layout()

    save_path = os.path.join(OUTPUTS_DIR, 'feature_importance.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Chart saved → {save_path}")


if __name__ == "__main__":
    main()