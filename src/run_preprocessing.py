import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config      import DATA_PROCESSED_DIR
from src.preprocessor import preprocess_dataframe, clean_text


def main():
    splits = ['train', 'val', 'test']

    for split in splits:
        path = os.path.join(DATA_PROCESSED_DIR, f"{split}.csv")
        print(f"\n{'='*45}")
        print(f"Processing: {split}.csv")
        print('='*45)

        df = pd.read_csv(path)
        preprocess_dataframe(df, text_col='text')

        # ── Show a before/after example ───────────────────
        if split == 'train':
            print("\n--- Before / After examples ---")
            for i in range(3):
                print(f"\nOriginal : {df['text'].iloc[i][:100]}")
                print(f"Cleaned  : {df['clean_text'].iloc[i][:100]}")

        # ── Save cleaned version ──────────────────────────
        out_path = os.path.join(DATA_PROCESSED_DIR, f"{split}_clean.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved → {out_path}")

    # ── Quick sanity check on a custom sentence ───────────
    print("\n" + "="*45)
    print("SANITY CHECK — custom sentences")
    print("="*45)

    test_sentences = [
        "I HATE those people!! Check http://example.com @user123 #racist",
        "RT @someone: This is a normal tweet about the weather :)",
        "You're not welcome here you disgusting &amp; horrible person!!!",
    ]

    for s in test_sentences:
        print(f"\nInput  : {s}")
        print(f"Output : {clean_text(s)}")


if __name__ == "__main__":
    main()