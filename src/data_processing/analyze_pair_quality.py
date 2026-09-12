import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spotify_customer_support_pairs.csv"
)


def main():
    print("=" * 70)
    print("SPOTIFY CUSTOMER SUPPORT PAIR QUALITY ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)

    print(f"\nTotal pairs: {len(df):,}")

    print("\n" + "=" * 70)
    print("MISSING VALUES")
    print("=" * 70)

    print(df.isnull().sum())

    print("\n" + "=" * 70)
    print("DUPLICATES")
    print("=" * 70)

    print(f"Duplicate rows: {df.duplicated().sum():,}")

    # Message lengths
    df["customer_length"] = df["customer_message"].astype(str).str.len()
    df["reply_length"] = df["spotify_reply"].astype(str).str.len()

    print("\n" + "=" * 70)
    print("MESSAGE LENGTH STATISTICS")
    print("=" * 70)

    print("\nCustomer message length:")
    print(df["customer_length"].describe())

    print("\nSpotify reply length:")
    print(df["reply_length"].describe())

    print("\n" + "=" * 70)
    print("SHORT MESSAGES")
    print("=" * 70)

    print(
        f"Customer messages under 10 characters: "
        f"{(df['customer_length'] < 10).sum():,}"
    )

    print(
        f"Spotify replies under 10 characters: "
        f"{(df['reply_length'] < 10).sum():,}"
    )

    print("\n" + "=" * 70)
    print("URL / MENTION ANALYSIS")
    print("=" * 70)

    print(
        f"Customer messages containing URLs: "
        f"{df['customer_message'].astype(str).str.contains('http').sum():,}"
    )

    print(
        f"Spotify replies containing URLs: "
        f"{df['spotify_reply'].astype(str).str.contains('http').sum():,}"
    )

    print(
        f"Customer messages containing @mentions: "
        f"{df['customer_message'].astype(str).str.contains('@').sum():,}"
    )

    print(
        f"Spotify replies containing @mentions: "
        f"{df['spotify_reply'].astype(str).str.contains('@').sum():,}"
    )

    print("\n" + "=" * 70)
    print("RANDOM SAMPLE")
    print("=" * 70)

    samples = df.sample(min(10, len(df)), random_state=42)

    for _, row in samples.iterrows():

        print("\nCUSTOMER:")
        print(row["customer_message"])

        print("\nSPOTIFY:")
        print(row["spotify_reply"])

        print("\n" + "-" * 70)


if __name__ == "__main__":
    main()