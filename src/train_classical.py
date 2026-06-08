"""
train_classical.py
------------------
Trains and evaluates traditional ML models:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Naive Bayes
  - Random Forest
  - Gradient Boosting

Uses both BoW and TF-IDF features.
Saves the best model for each feature type.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import load_npz
from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import LinearSVC
from sklearn.naive_bayes     import ComplementNB
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics         import (classification_report, confusion_matrix,
                                     f1_score, accuracy_score)
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODELS_DIR, OUTPUTS_DIR, LABEL_NAMES


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_labels():
    train = np.load(os.path.join(MODELS_DIR, 'train_labels.npy'))
    val   = np.load(os.path.join(MODELS_DIR, 'val_labels.npy'))
    test  = np.load(os.path.join(MODELS_DIR, 'test_labels.npy'))
    return train, val, test


def get_class_weights(labels):
    """
    Compute class weights to handle imbalance.
    Since hate speech is only 5.8% of data, we tell the model
    to penalize mistakes on that class more heavily.
    """
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return dict(zip(classes, weights))


def evaluate(model, X, y_true, split_name: str, label_names: dict):
    """Run predictions and print a full report."""
    y_pred = model.predict(X)

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='macro')
    f1_hate = f1_score(y_true, y_pred, average=None)[0]  # class 0 = hate speech

    print(f"\n  [{split_name}]  Accuracy: {acc:.4f}  |  "
          f"Macro-F1: {f1:.4f}  |  Hate-F1: {f1_hate:.4f}")
    print(classification_report(
        y_true, y_pred,
        target_names=[label_names[i] for i in sorted(label_names)]
    ))
    return y_pred, f1_hate


def plot_confusion_matrix(y_true, y_pred, label_names: dict,
                          title: str, save_path: str):
    cm = confusion_matrix(y_true, y_pred)
    # Normalize so each row sums to 1 (shows percentages)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names = [label_names[i] for i in sorted(label_names)]

    for ax, data, fmt, subtitle in zip(
        axes,
        [cm, cm_norm],
        ['d', '.2f'],
        ['Counts', 'Normalized']
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=names, yticklabels=names, ax=ax)
        ax.set_title(f'{title} — {subtitle}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════════════

def get_models(class_weights: dict):
    """
    Returns a dict of model_name → model instance.

    Notes on each:
    - LogisticRegression : fast, interpretable, strong baseline
    - LinearSVC          : best for high-dimensional text data
    - ComplementNB       : Naive Bayes variant best suited for imbalanced text
    - RandomForest       : ensemble of decision trees, handles non-linearity
    - GradientBoosting   : powerful but slow; we limit estimators for speed
    """
    return {
        'LogisticRegression': LogisticRegression(
            max_iter=1000,
            class_weight=class_weights,
            random_state=42,
            C=1.0
        ),
        'LinearSVC': LinearSVC(
            max_iter=2000,
            class_weight=class_weights,
            random_state=42,
            C=1.0
        ),
        'ComplementNB': ComplementNB(alpha=0.1),
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1      # use all CPU cores
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_on_feature_set(feature_name: str,
                         X_train, X_val, X_test,
                         y_train, y_val, y_test,
                         class_weights: dict):
    """Train all models on one feature set and return results."""

    print(f"\n{'█'*60}")
    print(f"  FEATURE SET: {feature_name}")
    print(f"  Train shape: {X_train.shape}")
    print(f"{'█'*60}")

    models   = get_models(class_weights)
    results  = []
    best_f1  = 0
    best_model_obj  = None
    best_model_name = None

    os.makedirs(os.path.join(OUTPUTS_DIR, 'confusion_matrices'), exist_ok=True)

    for model_name, model in models.items():
        # ComplementNB needs non-negative input — skip for TF-IDF with SVD
        # because SVD can produce negative values
        if model_name == 'ComplementNB' and 'tfidf' in feature_name:
            print(f"\n  Skipping ComplementNB for {feature_name} "
                  f"(SVD produces negative values)")
            continue

        print(f"\n  Training: {model_name} ...")
        model.fit(X_train, y_train)

        # Evaluate on validation set
        y_pred_val, f1_hate_val = evaluate(
            model, X_val, y_val, 'VAL', LABEL_NAMES
        )

        results.append({
            'feature'    : feature_name,
            'model'      : model_name,
            'val_hate_f1': round(f1_hate_val, 4),
            'val_macro_f1': round(
                f1_score(y_val, y_pred_val, average='macro'), 4
            )
        })

        # Track best model by hate speech F1
        if f1_hate_val > best_f1:
            best_f1         = f1_hate_val
            best_model_obj  = model
            best_model_name = model_name

    # ── Evaluate best model on test set ───────────────────────────────────
    print(f"\n  {'─'*50}")
    print(f"  Best model for {feature_name}: {best_model_name} "
          f"(Val Hate-F1: {best_f1:.4f})")
    print(f"  {'─'*50}")

    y_pred_test, _ = evaluate(
        best_model_obj, X_test, y_test, 'TEST', LABEL_NAMES
    )

    # Save confusion matrix
    plot_confusion_matrix(
        y_test, y_pred_test, LABEL_NAMES,
        title=f"{best_model_name} — {feature_name}",
        save_path=os.path.join(
            OUTPUTS_DIR, 'confusion_matrices',
            f"cm_{feature_name}_{best_model_name}.png"
        )
    )

    # Save best model
    model_path = os.path.join(
        MODELS_DIR, 'classifiers',
        f"best_{feature_name}_{best_model_name}.pkl"
    )
    with open(model_path, 'wb') as f:
        pickle.dump(best_model_obj, f)
    print(f"  Model saved → {model_path}")

    return results


def main():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # ── Load labels ────────────────────────────────────────────────────────
    y_train, y_val, y_test = load_labels()
    class_weights = get_class_weights(y_train)
    print(f"Class weights: { {LABEL_NAMES[k]: round(v,2) 
                               for k,v in class_weights.items()} }")

    all_results = []

    # ══════════════════════════════════════════════════════════════════════
    # 1. BOW FEATURES
    # ══════════════════════════════════════════════════════════════════════
    X_train_bow = load_npz(os.path.join(MODELS_DIR, 'X_train_bow.npz'))
    X_val_bow   = load_npz(os.path.join(MODELS_DIR, 'X_val_bow.npz'))
    X_test_bow  = load_npz(os.path.join(MODELS_DIR, 'X_test_bow.npz'))

    results_bow = train_on_feature_set(
        'bow',
        X_train_bow, X_val_bow, X_test_bow,
        y_train, y_val, y_test,
        class_weights
    )
    all_results.extend(results_bow)

    # ══════════════════════════════════════════════════════════════════════
    # 2. TF-IDF FEATURES  (all three scales)
    # ══════════════════════════════════════════════════════════════════════
    for n in [512, 1024, 2048]:
        X_train = np.load(os.path.join(MODELS_DIR, f'X_train_tfidf_{n}.npy'))
        X_val   = np.load(os.path.join(MODELS_DIR, f'X_val_tfidf_{n}.npy'))
        X_test  = np.load(os.path.join(MODELS_DIR, f'X_test_tfidf_{n}.npy'))

        results = train_on_feature_set(
            f'tfidf_{n}',
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            class_weights
        )
        all_results.extend(results)

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "═"*65)
    print("SUMMARY — All models ranked by Hate Speech F1 (Validation)")
    print("═"*65)

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('val_hate_f1', ascending=False)
    print(results_df.to_string(index=False))

    # Save summary
    summary_path = os.path.join(OUTPUTS_DIR, 'classical_ml_results.csv')
    results_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved → {summary_path}")

    # ── Bar chart of results ───────────────────────────────────────────────
    plt.figure(figsize=(14, 6))
    labels_chart = [f"{r['model']}\n({r['feature']})"
                    for _, r in results_df.iterrows()]
    plt.bar(range(len(results_df)),
            results_df['val_hate_f1'],
            color='steelblue', edgecolor='black')
    plt.xticks(range(len(results_df)), labels_chart,
               rotation=45, ha='right', fontsize=8)
    plt.ylabel('Hate Speech F1 Score')
    plt.title('Classical ML Models — Hate Speech F1 on Validation Set')
    plt.axhline(y=0.85, color='red', linestyle='--',
                label='Target F1 = 0.85')
    plt.legend()
    plt.tight_layout()
    chart_path = os.path.join(OUTPUTS_DIR, 'classical_ml_comparison.png')
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Chart saved → {chart_path}")


if __name__ == "__main__":
    main()