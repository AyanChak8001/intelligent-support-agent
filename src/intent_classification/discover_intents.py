import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SAMPLE_SIZE = 200
RANDOM_SEED = 42

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spotify_customer_support_pairs_clean.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spotify_intent_discovery_sample.csv"
)


def main():

    print("=" * 70)
    print("SPOTIFY INTENT DISCOVERY")
    print("=" * 70)

    df = pd.read_csv(INPUT_PATH)

    print(f"\nTotal clean conversations: {len(df):,}")

    # Take a reproducible random sample
    if len(df) < SAMPLE_SIZE:
        raise ValueError(
            f"Need at least {SAMPLE_SIZE} clean conversations; found {len(df)}."
        )

    sample_size = SAMPLE_SIZE

    sample = df.sample(
        n=sample_size,
        random_state=RANDOM_SEED
    ).copy()

    # Keep the most useful columns for manual analysis
    sample = sample[
        [
            "customer_tweet_id",
            "customer_message_clean",
            "spotify_reply_clean",
        ]
    ]

    # Add empty columns for manual labeling later
    sample["candidate_intent"] = ""
    sample["notes"] = ""
    sample["sample_seed"] = RANDOM_SEED

    # Save sample
    sample.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSample size: {len(sample):,}")

    print("\nSample saved to:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 70)
    print("FIRST 20 CUSTOMER MESSAGES")
    print("=" * 70)

    for i, (_, row) in enumerate(sample.head(20).iterrows(), start=1):

        print(f"\n{i}. {row['customer_message_clean']}")


if __name__ == "__main__":
    main()
