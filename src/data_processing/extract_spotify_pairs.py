import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "twcs.csv"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spotify_customer_support_pairs.csv"
)

BRAND = "SpotifyCares"


def main():
    print("=" * 70)
    print(f"EXTRACTING CUSTOMER → {BRAND} SUPPORT PAIRS")
    print("=" * 70)

    # Load required columns
    df = pd.read_csv(
        DATA_PATH,
        usecols=[
            "tweet_id",
            "author_id",
            "inbound",
            "created_at",
            "text",
            "response_tweet_id",
            "in_response_to_tweet_id",
        ],
    )

    print(f"\nTotal tweets loaded: {len(df):,}")

    # Find replies sent by Spotify
    spotify_replies = df[
        (df["author_id"] == BRAND)
        & (df["inbound"] == False)
    ].copy()

    print(f"Spotify support replies: {len(spotify_replies):,}")

    # Keep only replies that directly respond to another tweet
    spotify_replies = spotify_replies.dropna(
        subset=["in_response_to_tweet_id"]
    )

    print(
        f"Replies linked to a previous tweet: "
        f"{len(spotify_replies):,}"
    )

    # Rename columns before joining
    spotify_replies = spotify_replies.rename(
        columns={
            "tweet_id": "spotify_reply_id",
            "text": "spotify_reply",
            "created_at": "reply_created_at",
            "in_response_to_tweet_id": "customer_tweet_id",
        }
    )

    # Customer tweets are inbound messages
    customer_messages = df[df["inbound"] == True].copy()

    customer_messages = customer_messages.rename(
        columns={
            "tweet_id": "customer_tweet_id",
            "author_id": "customer_id",
            "text": "customer_message",
            "created_at": "customer_created_at",
        }
    )

    # Join customer messages with Spotify replies
    pairs = spotify_replies.merge(
        customer_messages[
            [
                "customer_tweet_id",
                "customer_id",
                "customer_message",
                "customer_created_at",
            ]
        ],
        on="customer_tweet_id",
        how="inner",
    )

    # Select useful columns
    pairs = pairs[
        [
            "customer_tweet_id",
            "customer_id",
            "customer_message",
            "customer_created_at",
            "spotify_reply_id",
            "spotify_reply",
            "reply_created_at",
        ]
    ]

    # Remove missing text
    pairs = pairs.dropna(
        subset=["customer_message", "spotify_reply"]
    )

    # Remove duplicates
    pairs = pairs.drop_duplicates(
        subset=["customer_tweet_id", "spotify_reply_id"]
    )

    print(f"\nClean customer → Spotify pairs: {len(pairs):,}")

    # Show examples
    print("\n" + "=" * 70)
    print("SAMPLE CONVERSATIONS")
    print("=" * 70)

    for _, row in pairs.head(5).iterrows():
        print("\nCUSTOMER:")
        print(row["customer_message"])

        print("\nSPOTIFY:")
        print(row["spotify_reply"])

        print("\n" + "-" * 70)

    # Save processed dataset
    pairs.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved processed dataset to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()