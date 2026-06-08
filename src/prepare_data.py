import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (DATA_RAW_DIR, DATA_PROCESSED_DIR,
                        RANDOM_SEED, TEST_SIZE, VAL_SIZE, LABEL_NAMES)

def prepare_data():
    # ── Load raw data ──────────────────────────────────────
    filepath = os.path.join(DATA_RAW_DIR, "labeled_data.csv")
    df = pd.read_csv(filepath)

    print(f"Loaded {len(df)} samples")

    # ── Keep only what we need ─────────────────────────────
    df = df[['tweet', 'class']].copy()
    df.columns = ['text', 'label']          # rename for clarity

    # ── Drop any missing values ────────────────────────────
    before = len(df)
    df.dropna(inplace=True)
    print(f"Dropped {before - len(df)} rows with missing values")

    # ── Remove duplicates ──────────────────────────────────
    before = len(df)
    df.drop_duplicates(subset='text', inplace=True)
    print(f"Dropped {before - len(df)} duplicate rows")

    print(f"\nFinal dataset size: {len(df)} samples")

    # ── Split: first separate test set ────────────────────
    # stratify=df['label'] ensures each split has same class ratio
    df_train_val, df_test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df['label']
    )

    # ── Split: then separate validation from training ──────
    # val is 15% of total, which is 15/85 of the remaining train_val
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_ratio,
        random_state=RANDOM_SEED,
        stratify=df_train_val['label']
    )

    print(f"\nTrain size      : {len(df_train)} samples")
    print(f"Validation size : {len(df_val)} samples")
    print(f"Test size       : {len(df_test)} samples")

    # ── Verify class balance in each split ─────────────────
    print("\n--- Class distribution in each split ---")
    for split_name, split_df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        print(f"\n{split_name}:")
        counts = split_df['label'].value_counts().sort_index()
        for label_id, count in counts.items():
            pct = count / len(split_df) * 100
            print(f"  {LABEL_NAMES[label_id]:25s}: {count:5d} ({pct:.1f}%)")

    # ── Save to processed folder ───────────────────────────
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

    df_train.to_csv(os.path.join(DATA_PROCESSED_DIR, "train.csv"), index=False)
    df_val.to_csv(os.path.join(DATA_PROCESSED_DIR, "val.csv"),   index=False)
    df_test.to_csv(os.path.join(DATA_PROCESSED_DIR, "test.csv"), index=False)
    df.to_csv(os.path.join(DATA_PROCESSED_DIR, "full.csv"),      index=False)

    print(f"\nSaved to: {DATA_PROCESSED_DIR}")
    print("  train.csv, val.csv, test.csv, full.csv")

if __name__ == "__main__":
    prepare_data()