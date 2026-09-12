import pandas as pd
from pathlib import Path
import re
import html


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spotify_customer_support_pairs.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spotify_customer_support_pairs_clean.csv"
)


def clean_text(text):
    """
    Clean Twitter-specific noise while preserving
    the actual meaning of the support conversation.
    """

    # Convert to string
    text = str(text)

    # Decode HTML entities: &amp; -> &
    text = html.unescape(text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove Twitter mentions
    text = re.sub(r"@\w+", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():

    print("=" * 70)
    print("CLEANING SPOTIFY CUSTOMER SUPPORT DATA")
    print("=" * 70)

    # Load extracted pairs
    df = pd.read_csv(INPUT_PATH)

    print(f"\nOriginal pairs: {len(df):,}")

    # Clean customer messages
    df["customer_message_clean"] = (
        df["customer_message"]
        .apply(clean_text)
    )

    # Clean Spotify replies
    df["spotify_reply_clean"] = (
        df["spotify_reply"]
        .apply(clean_text)
    )

    # Remove very short messages
    before = len(df)

    df = df[
        (df["customer_message_clean"].str.len() >= 10)
        &
        (df["spotify_reply_clean"].str.len() >= 10)
    ].copy()

    removed = before - len(df)

    print(f"Removed short examples: {removed:,}")
    print(f"Remaining pairs: {len(df):,}")

    # Remove duplicates after cleaning
    before_duplicates = len(df)

    df = df.drop_duplicates(
        subset=[
            "customer_message_clean",
            "spotify_reply_clean"
        ]
    )

    duplicates_removed = before_duplicates - len(df)

    print(
        f"Duplicates removed after cleaning: "
        f"{duplicates_removed:,}"
    )

    print(f"Final clean dataset: {len(df):,}")

    # Show sample before/after cleaning
    print("\n" + "=" * 70)
    print("BEFORE / AFTER EXAMPLES")
    print("=" * 70)

    samples = df.sample(
        min(5, len(df)),
        random_state=42
    )

    for _, row in samples.iterrows():

        print("\nORIGINAL CUSTOMER:")
        print(row["customer_message"])

        print("\nCLEAN CUSTOMER:")
        print(row["customer_message_clean"])

        print("\nORIGINAL SPOTIFY:")
        print(row["spotify_reply"])

        print("\nCLEAN SPOTIFY:")
        print(row["spotify_reply_clean"])

        print("\n" + "-" * 70)

    # Save clean dataset
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 70)
    print("CLEAN DATASET SAVED")
    print("=" * 70)

    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()