"""Safe majority-baseline assessment.

The independent majority-class path requires a separate labeled source and
never uses golden final_intent values. A separate explicitly named helper
provides only a descriptive, label-leaking sanity check when requested.
"""

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "results" / "golden_evaluation_dataset.csv"
DEFAULT_SOURCES = [
    ROOT / "data" / "evaluation" / "labeled_intent_sample.csv",
    ROOT / "data" / "evaluation" / "intent_labeling_progress.csv",
]
TAXONOMY = {
    "APP_TECHNICAL_COMPATIBILITY", "SUBSCRIPTION_PLAN", "BILLING_PAYMENT_REFUND",
    "PLAYLIST_LIBRARY_DISCOVERY", "DOWNLOADS_OFFLINE_STORAGE", "ACCOUNT_ACCESS_SECURITY",
    "PLAYBACK_AUDIO", "COMMUNITY_INTEGRATIONS", "CONTENT_AVAILABILITY_METADATA",
    "FEATURE_REQUEST_PRODUCT_FEEDBACK", "OTHER_UNCLEAR",
}
TAXONOMY_ORDER = [
    "APP_TECHNICAL_COMPATIBILITY", "SUBSCRIPTION_PLAN", "BILLING_PAYMENT_REFUND",
    "PLAYLIST_LIBRARY_DISCOVERY", "DOWNLOADS_OFFLINE_STORAGE", "ACCOUNT_ACCESS_SECURITY",
    "PLAYBACK_AUDIO", "COMMUNITY_INTEGRATIONS", "CONTENT_AVAILABILITY_METADATA",
    "FEATURE_REQUEST_PRODUCT_FEEDBACK", "OTHER_UNCLEAR",
]


def _golden_ids():
    golden = pd.read_csv(GOLDEN, keep_default_na=False)
    return set(golden["example_id"])


def _source_ids(frame):
    if "example_id" in frame:
        return set(frame["example_id"])
    if "customer_tweet_id" in frame:
        return {f"spotify-{int(float(value))}" for value in frame["customer_tweet_id"]}
    return set()


def assess_majority(sources=None):
    golden_ids = _golden_ids()
    candidates = []
    for path in sources or DEFAULT_SOURCES:
        if not path.exists():
            continue
        frame = pd.read_csv(path, keep_default_na=False)
        label_column = "final_intent" if "final_intent" in frame else "intent" if "intent" in frame else None
        if label_column is None:
            continue
        labeled = frame[frame[label_column].isin(TAXONOMY)].copy()
        if "example_id" in labeled:
            eligible = labeled.loc[~labeled["example_id"].isin(golden_ids)]
        elif "customer_tweet_id" in labeled:
            row_ids = labeled["customer_tweet_id"].map(lambda value: f"spotify-{int(float(value))}")
            eligible = labeled.loc[~row_ids.isin(golden_ids)]
        else:
            eligible = labeled.iloc[0:0]
        candidates.append({"path": str(path), "labeled_rows": len(labeled), "eligible_rows": len(eligible)})
        if len(eligible):
            counts = eligible[label_column].value_counts()
            return {"available": True, "source": str(path), "label_column": label_column, "eligible_rows": len(eligible), "majority_intent": counts.index[0], "class_counts": counts.to_dict(), "candidates": candidates}
    return {"available": False, "reason": "No separate labeled rows remain after excluding golden evaluation IDs.", "candidates": candidates}


def get_majority_intent(sources=None):
    """Return the separate-source majority intent, or None if unavailable."""
    result = assess_majority(sources)
    return result.get("majority_intent") if result["available"] else None


def descriptive_majority_intent(labels):
    """Return the most frequent evaluation label for a sanity check only.

    This is intentionally not an independent baseline: the same evaluation
    labels select the constant class and assess the prediction. Ties follow
    the fixed taxonomy order.
    """
    counts = pd.Series(labels).value_counts()
    if counts.empty:
        raise ValueError("Cannot select a majority class from empty labels.")
    majority = max(TAXONOMY_ORDER, key=lambda intent: (int(counts.get(intent, 0)), -TAXONOMY_ORDER.index(intent)))
    return majority, {intent: int(counts.get(intent, 0)) for intent in TAXONOMY_ORDER}


def predict(_customer_message=None, _spotify_reply=None, sources=None):
    """Always predict the separate-source majority class when one exists."""
    intent = get_majority_intent(sources)
    if intent is None:
        raise RuntimeError("Majority baseline unavailable: no eligible separate labeled source data.")
    return intent


def main():
    result = assess_majority()
    output = ROOT / "results" / "majority_baseline_status.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if result["available"]:
        print(f"Majority baseline available: {result['majority_intent']}")
    else:
        print("Majority baseline unavailable: no eligible separate labeled training rows.")
    print(output)


if __name__ == "__main__":
    main()
