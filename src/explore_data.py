import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add project root to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_RAW_DIR, OUTPUTS_DIR, LABEL_NAMES

def explore_dataset():
    # ── Load the CSV ───────────────────────────────────────
    filepath = os.path.join(DATA_RAW_DIR, "labeled_data.csv")
    df = pd.read_csv(filepath)

    print("=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)

    # Shape = (rows, columns)
    print(f"\nTotal samples : {df.shape[0]}")
    print(f"Total columns : {df.shape[1]}")

    print("\n--- First 5 rows ---")
    print(df.head())

    print("\n--- Column names ---")
    print(df.columns.tolist())

    print("\n--- Data types ---")
    print(df.dtypes)

    print("\n--- Missing values ---")
    print(df.isnull().sum())

    # ── Class distribution ─────────────────────────────────
    # 'class' column: 0=hate, 1=offensive, 2=neither
    print("\n--- Class Distribution ---")
    class_counts = df['class'].value_counts().sort_index()
    for label_id, count in class_counts.items():
        label_name = LABEL_NAMES[label_id]
        percentage  = (count / len(df)) * 100
        print(f"  {label_name:25s}: {count:6d} ({percentage:.1f}%)")

    # ── Sample tweets from each class ─────────────────────
    print("\n--- Sample tweets from each class ---")
    for label_id, label_name in LABEL_NAMES.items():
        print(f"\n[{label_name}]")
        samples = df[df['class'] == label_id]['tweet'].sample(3, random_state=42)
        for i, tweet in enumerate(samples, 1):
            print(f"  {i}. {tweet[:120]}")

    # ── Tweet length stats ─────────────────────────────────
    df['tweet_length'] = df['tweet'].apply(len)
    print("\n--- Tweet Length Statistics ---")
    print(df['tweet_length'].describe())

    # ── Save class distribution chart ─────────────────────
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    plt.figure(figsize=(8, 5))
    colors = ['#e74c3c', '#e67e22', '#2ecc71']
    bars = plt.bar(
        [LABEL_NAMES[i] for i in range(3)],
        [class_counts[i] for i in range(3)],
        color=colors,
        edgecolor='black'
    )
    # Add count labels on top of each bar
    for bar, count in zip(bars, [class_counts[i] for i in range(3)]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 100,
            str(count),
            ha='center', fontsize=11, fontweight='bold'
        )
    plt.title('Class Distribution — Davidson Dataset', fontsize=14)
    plt.ylabel('Number of Tweets')
    plt.tight_layout()

    chart_path = os.path.join(OUTPUTS_DIR, "class_distribution.png")
    plt.savefig(chart_path)
    print(f"\nChart saved to: {chart_path}")
    plt.show()

    return df

if __name__ == "__main__":
    df = explore_dataset()