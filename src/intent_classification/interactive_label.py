"""Interactive human review for automatic intent suggestions.

Automatic suggestions and human decisions are stored in separate columns. The
script never creates the golden dataset automatically.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT_DIR / "data" / "evaluation" / "intent_suggestions_all_200.csv"
QUEUE_FILE = ROOT_DIR / "data" / "evaluation" / "manual_review_queue.csv"
PROGRESS_FILE = ROOT_DIR / "data" / "evaluation" / "intent_suggestions_review_progress.csv"
METADATA_FILE = ROOT_DIR / "results" / "golden_evaluation_metadata.json"
SAMPLE_SIZE = 200
INTENTS = {
    "1": "APP_TECHNICAL_COMPATIBILITY", "2": "SUBSCRIPTION_PLAN",
    "3": "BILLING_PAYMENT_REFUND", "4": "PLAYLIST_LIBRARY_DISCOVERY",
    "5": "DOWNLOADS_OFFLINE_STORAGE", "6": "ACCOUNT_ACCESS_SECURITY",
    "7": "PLAYBACK_AUDIO", "8": "COMMUNITY_INTEGRATIONS",
    "9": "CONTENT_AVAILABILITY_METADATA", "10": "FEATURE_REQUEST_PRODUCT_FEEDBACK",
    "11": "OTHER_UNCLEAR",
}

def _ensure_columns(df):
    defaults = {"final_intent": "", "review_status": "PENDING", "review_notes": "", "reviewed": False, "label_source": "", "label_action": ""}
    for column, default in defaults.items():
        if column not in df:
            df[column] = default
        if column != "reviewed":
            df[column] = df[column].fillna("").astype(object)
    df["reviewed"] = df["reviewed"].fillna(False).astype(bool)
    return df

def load_progress():
    source = PROGRESS_FILE if PROGRESS_FILE.exists() else INPUT_FILE
    if not source.exists():
        raise FileNotFoundError(f"Input file not found: {source}")
    df = _ensure_columns(pd.read_csv(source, keep_default_na=False))
    if len(df) != SAMPLE_SIZE:
        raise ValueError(f"Expected exactly {SAMPLE_SIZE} examples, found {len(df)} in {source}.")
    return df

def save_progress(df):
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROGRESS_FILE, index=False)

def save_metadata(df, created_at):
    reviewed = df["review_status"].eq("REVIEWED")
    metadata = {
        "dataset_name": "golden_evaluation_dataset",
        "sample_size": int(len(df)), "created_at_utc": created_at,
        "number_reviewed_examples": int(reviewed.sum()),
        "number_human_approved": int((df["label_action"] == "human_approved_suggestion").sum()),
        "number_human_overridden": int((df["label_action"] == "human_override").sum()),
        "golden_dataset_ready": False,
        "labeling_note": "Automatic suggestions are not human labels; this workflow never writes golden_evaluation_dataset.csv automatically.",
    }
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata

def print_intents():
    print("\nAVAILABLE INTENTS:")
    for number, intent in INTENTS.items():
        print(f"{number}. {intent}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-flagged", action="store_true", help="Review only rows with requires_human_review=TRUE.")
    args = parser.parse_args()
    try:
        df = load_progress()
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}")
        return
    created_at = datetime.now(timezone.utc).isoformat()
    flagged = df["requires_human_review"].astype(str).str.upper().eq("TRUE") if "requires_human_review" in df else pd.Series(False, index=df.index)
    pending_mask = df["review_status"].ne("REVIEWED")
    if args.review_flagged:
        pending_mask &= flagged
    pending = df.index[pending_mask].tolist()
    print("INTERACTIVE INTENT LABELING")
    print(f"Mode: {'flagged cases only' if args.review_flagged else 'full review'}")
    print(f"Total examples: {len(df)} | Awaiting review in this mode: {len(pending)}")
    print("Enter = approve suggestion, 1-11 = override, s = skip, q = save and quit")
    print_intents()
    for index in pending:
        row = df.loc[index]
        print("\n" + "=" * 70)
        print(f"EXAMPLE {index + 1}/{len(df)}")
        print("CUSTOMER MESSAGE:\n" + str(row.get("customer_message", row.get("customer_message_clean", ""))))
        print("\nHISTORICAL SPOTIFY REPLY:\n" + str(row.get("spotify_reply", row.get("spotify_reply_clean", ""))))
        print(f"\nSUGGESTED INTENT: {row.get('suggested_intent', 'N/A')}")
        print(f"CONFIDENCE: {row.get('confidence', row.get('suggested_confidence', 'N/A'))}")
        print(f"REASON: {row.get('confidence_reason', '')}")
        print(f"SOURCE: {row.get('suggestion_source', 'N/A')}")
        while True:
            choice = input("\nChoose (Enter/1-11/s/q): ").strip().lower()
            if choice == "q":
                save_progress(df); save_metadata(df, created_at)
                print(f"Progress saved to {PROGRESS_FILE}"); return
            if choice == "s":
                df.at[index, "review_status"] = "SKIPPED"; df.at[index, "reviewed"] = False
                save_progress(df); save_metadata(df, created_at); break
            if choice == "":
                selected = str(row.get("suggested_intent", ""))
                if selected not in INTENTS.values():
                    print("No valid suggestion; choose 1-11."); continue
                action = "human_approved_suggestion"
            elif choice in INTENTS:
                selected = INTENTS[choice]; action = "human_override"
            else:
                print("Invalid choice. Enter, 1-11, s, or q."); continue
            df.at[index, "final_intent"] = selected
            df.at[index, "review_status"] = "REVIEWED"
            df.at[index, "reviewed"] = True
            df.at[index, "label_source"] = "human"
            df.at[index, "label_action"] = action
            save_progress(df); save_metadata(df, created_at)
            print(f"Saved human label: {selected} ({action})")
            break
    save_progress(df); metadata = save_metadata(df, created_at)
    print(f"\nReview session complete. Reviewed in file: {metadata['number_reviewed_examples']}")
    print("No golden dataset was generated; export only after a separate human-labeling decision.")

if __name__ == "__main__":
    main()
