"""
prepare_conan.py
----------------
Loads the CONAN dataset (hate speech → counterspeech pairs)
and prepares it for fine-tuning DialoGPT.

CONAN contains expert-written responses to hate speech across
topics like Islamophobia, antisemitism, and more.
"""

import json
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, RANDOM_SEED


def prepare_conan():
    conan_path = os.path.join(DATA_RAW_DIR, 'CONAN.json')

    if not os.path.exists(conan_path):
        print("ERROR: CONAN.json not found in data/raw/")
        print("Run: wget https://raw.githubusercontent.com/"
              "marcoguerini/CONAN/master/CONAN/CONAN.json")
        return

    print("Loading CONAN dataset...")
    with open(conan_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # ── Parse the JSON structure ───────────────────────────────────────
    # CONAN has a 'conan' key containing list of {hateSpeech, counterSpeech}
    pairs = []

    # Handle different possible JSON structures
    if isinstance(data, dict) and 'conan' in data:
        entries = data['conan']
    elif isinstance(data, list):
        entries = data
    else:
        # Try to find pairs in nested structure
        entries = []
        for key, value in data.items():
            if isinstance(value, list):
                entries.extend(value)

    print(f"Raw entries found: {len(entries)}")

    for entry in entries:
        # Different versions of CONAN use different key names
        hate = (entry.get('hateSpeech') or
                entry.get('hate_speech') or
                entry.get('HS') or '')
        counter = (entry.get('counterSpeech') or
                   entry.get('counter_speech') or
                   entry.get('CN') or '')

        if hate and counter:
            pairs.append({
                'hate_speech'   : hate.strip(),
                'counter_speech': counter.strip()
            })

    print(f"Valid pairs extracted: {len(pairs)}")

    if len(pairs) == 0:
        print("\nDEBUG — showing raw JSON structure:")
        print(json.dumps(data if isinstance(data, dict)
                         else data[0], indent=2)[:500])
        return

    df = pd.DataFrame(pairs)

    # ── Basic stats ────────────────────────────────────────────────────
    print(f"\nHate speech avg length   : "
          f"{df['hate_speech'].str.len().mean():.0f} chars")
    print(f"Counter speech avg length: "
          f"{df['counter_speech'].str.len().mean():.0f} chars")

    print("\nSample pairs:")
    for i in range(min(3, len(df))):
        print(f"\n  Hate    : {df['hate_speech'].iloc[i][:100]}")
        print(f"  Counter : {df['counter_speech'].iloc[i][:100]}")

    # ── Split ──────────────────────────────────────────────────────────
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=RANDOM_SEED
    )

    print(f"\nSplits — Train: {len(train_df)} | "
          f"Val: {len(val_df)} | Test: {len(test_df)}")

    # ── Save ───────────────────────────────────────────────────────────
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

    train_df.to_csv(
        os.path.join(DATA_PROCESSED_DIR, 'conan_train.csv'), index=False)
    val_df.to_csv(
        os.path.join(DATA_PROCESSED_DIR, 'conan_val.csv'),   index=False)
    test_df.to_csv(
        os.path.join(DATA_PROCESSED_DIR, 'conan_test.csv'),  index=False)

    print(f"\nSaved to {DATA_PROCESSED_DIR}")
    print("  conan_train.csv, conan_val.csv, conan_test.csv")

    return df


if __name__ == "__main__":
    prepare_conan()