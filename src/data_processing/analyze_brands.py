import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "twcs.csv"


def main():
    print("=" * 70)
    print("ANALYZING BRANDS")
    print("=" * 70)

    # Load only columns we need
    df = pd.read_csv(
        DATA_PATH,
        usecols=[
            "tweet_id",
            "author_id",
            "inbound",
            "text",
            "response_tweet_id",
            "in_response_to_tweet_id"
        ]
    )

    # Brand/support messages are outbound
    brand_messages = df[df["inbound"] == False]

    print(f"\nTotal tweets: {len(df):,}")
    print(f"Customer messages: {(df['inbound'] == True).sum():,}")
    print(f"Brand/support messages: {len(brand_messages):,}")

    # Count messages sent by each brand
    brand_counts = (
        brand_messages["author_id"]
        .value_counts()
        .reset_index()
    )

    brand_counts.columns = ["brand", "support_messages"]

    print("\n" + "=" * 70)
    print("TOP 30 BRANDS BY SUPPORT MESSAGES")
    print("=" * 70)

    print(brand_counts.head(30).to_string(index=False))

    # Save results so we can inspect them later
    output_path = PROJECT_ROOT / "results" / "brand_analysis.csv"

    brand_counts.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()