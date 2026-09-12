import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "intent_review_with_suggestions.csv"
)


def main():

    df = pd.read_csv(INPUT_PATH)

    unclear_df = df[
        df["suggested_intent"] == "OTHER_UNCLEAR"
    ].copy()

    print("=" * 70)
    print("OTHER / UNCLEAR MESSAGE ANALYSIS")
    print("=" * 70)

    print(f"\nTotal OTHER_UNCLEAR messages: {len(unclear_df)}")

    print("\n" + "=" * 70)
    print("FIRST 50 OTHER_UNCLEAR MESSAGES")
    print("=" * 70)

    for i, (_, row) in enumerate(
        unclear_df.head(50).iterrows(),
        start=1
    ):

        print(f"\n{i}. {row['customer_message_clean']}")


if __name__ == "__main__":
    main()