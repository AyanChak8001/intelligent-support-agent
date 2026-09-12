import pandas as pd
from pathlib import Path


# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset path
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "twcs.csv"


def main():
    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)

    print(f"\nDataset shape: {df.shape}")
    print(f"Total rows: {len(df):,}")

    print("\n" + "=" * 60)
    print("COLUMNS")
    print("=" * 60)

    print(df.columns.tolist())

    print("\n" + "=" * 60)
    print("FIRST 5 ROWS")
    print("=" * 60)

    print(df.head())

    print("\n" + "=" * 60)
    print("DATA TYPES")
    print("=" * 60)

    print(df.dtypes)

    print("\n" + "=" * 60)
    print("TOP BRANDS BY NUMBER OF TWEETS")
    print("=" * 60)

    brand_counts = df["author_id"].value_counts()

    print(brand_counts.head(30))

    print("\n" + "=" * 60)
    print("SAMPLE BRAND CONVERSATIONS")
    print("=" * 60)

    print(df[df["author_id"] == "AmazonHelp"].head(10))


if __name__ == "__main__":
    main()