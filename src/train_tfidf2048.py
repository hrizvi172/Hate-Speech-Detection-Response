"""
train_tfidf2048.py
------------------
Runs training only for the missing tfidf_2048 feature set.
"""

import os, sys, numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODELS_DIR, OUTPUTS_DIR
from src.train_classical import (load_labels, get_class_weights,
                                  train_on_feature_set)
import pandas as pd

def main():
    y_train, y_val, y_test = load_labels()
    class_weights = get_class_weights(y_train)

    X_train = np.load(os.path.join(MODELS_DIR, 'X_train_tfidf_2048.npy'))
    X_val   = np.load(os.path.join(MODELS_DIR, 'X_val_tfidf_2048.npy'))
    X_test  = np.load(os.path.join(MODELS_DIR, 'X_test_tfidf_2048.npy'))

    results = train_on_feature_set(
        'tfidf_2048',
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        class_weights
    )

    # Append to existing results CSV
    results_df   = pd.DataFrame(results)
    existing_path = os.path.join(OUTPUTS_DIR, 'classical_ml_results.csv')

    if os.path.exists(existing_path):
        existing = pd.read_csv(existing_path)
        combined = pd.concat([existing, results_df], ignore_index=True)
    else:
        combined = results_df

    combined = combined.sort_values('val_hate_f1', ascending=False)
    combined.to_csv(existing_path, index=False)

    print('\n' + '='*55)
    print('FINAL SUMMARY — All results')
    print('='*55)
    print(combined.to_string(index=False))

if __name__ == "__main__":
    main()