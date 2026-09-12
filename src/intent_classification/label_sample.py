import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT_DIR / "data" / "evaluation" / "intent_review_with_suggestions.csv"

OUTPUT_FILE = ROOT_DIR / "data" / "evaluation" / "intent_labeling_progress.csv"


# ============================================================
# FINAL INTENT TAXONOMY
# ============================================================

VALID_INTENTS = [
    "APP_TECHNICAL_COMPATIBILITY",
    "SUBSCRIPTION_PLAN",
    "BILLING_PAYMENT_REFUND",
    "PLAYLIST_LIBRARY_DISCOVERY",
    "DOWNLOADS_OFFLINE_STORAGE",
    "ACCOUNT_ACCESS_SECURITY",
    "PLAYBACK_AUDIO",
    "COMMUNITY_INTEGRATIONS",
    "CONTENT_AVAILABILITY_METADATA",
    "FEATURE_REQUEST_PRODUCT_FEEDBACK",
    "OTHER_UNCLEAR",
]


def main():

    print("=" * 70)
    print("CREATING INTENT LABELING DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        print(f"\nERROR: Input file not found:")
        print(INPUT_FILE)
        return

    df = pd.read_csv(INPUT_FILE)

    print(f"\nMessages loaded: {len(df)}")

    print("\nAvailable columns:")
    print(list(df.columns))

    # --------------------------------------------------------
    # CREATE LABELING COLUMNS
    # --------------------------------------------------------

    # Create final_intent if it doesn't already exist
    if "final_intent" not in df.columns:
        df["final_intent"] = ""

    # Create review status
    if "review_status" not in df.columns:
        df["review_status"] = "PENDING"

    # Create reviewer notes
    if "review_notes" not in df.columns:
        df["review_notes"] = ""

    if "reviewed" not in df.columns:
        df["reviewed"] = False

    if "label_source" not in df.columns:
        df["label_source"] = ""

    # --------------------------------------------------------
    # REORDER COLUMNS
    # --------------------------------------------------------

    preferred_columns = []

    if "customer_message" in df.columns:
        preferred_columns.append("customer_message")

    if "suggested_intent" in df.columns:
        preferred_columns.append("suggested_intent")

    preferred_columns.extend([
        "final_intent",
        "review_status",
        "review_notes"
    ])

    # Keep any remaining columns
    remaining_columns = [
        col for col in df.columns
        if col not in preferred_columns
    ]

    df = df[preferred_columns + remaining_columns]

    # --------------------------------------------------------
    # SAVE FILE
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 70)
    print("LABELING FILE CREATED SUCCESSFULLY")
    print("=" * 70)

    print(f"\nSaved to:")
    print(OUTPUT_FILE)

    print(f"\nTotal messages: {len(df)}")

    print("\nFINAL INTENTS:")
    for i, intent in enumerate(VALID_INTENTS, start=1):
        print(f"{i}. {intent}")

    print("\nNEXT ACTION:")
    print("Review each customer message and fill:")
    print("- final_intent")
    print("- review_status")
    print("- review_notes (optional)")


if __name__ == "__main__":
    main()
