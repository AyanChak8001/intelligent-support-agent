import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spotify_intent_discovery_sample.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "intent_review_sample.csv"
)


def main():

    df = pd.read_csv(INPUT_PATH)

    # Add clearer review columns
    review_df = df[
        [
            "customer_tweet_id",
            "customer_message_clean",
            "spotify_reply_clean",
        ]
    ].copy()

    review_df["intent"] = ""
    review_df["is_clear_standalone_issue"] = ""
    review_df["notes"] = ""

    review_df.to_csv(OUTPUT_PATH, index=False)

    print("=" * 60)
    print("INTENT REVIEW FILE CREATED")
    print("=" * 60)

    print(f"\nTotal messages for review: {len(review_df)}")
    print(f"\nSaved to:\n{OUTPUT_PATH}")

    print("\nReview the file and identify recurring patterns.")
    print("Do NOT label everything yet.")


if __name__ == "__main__":
    main()