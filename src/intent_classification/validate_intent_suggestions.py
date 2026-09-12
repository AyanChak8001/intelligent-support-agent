"""Validate and report the automatic 200-example intent suggestion artifacts."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SUGGESTIONS = ROOT / "data" / "evaluation" / "intent_suggestions_all_200.csv"
QUEUE = ROOT / "data" / "evaluation" / "manual_review_queue.csv"
PROGRESS = ROOT / "data" / "evaluation" / "intent_suggestions_review_progress.csv"
BEFORE = ROOT / "data" / "evaluation" / "intent_suggestions_before_improvement_200.csv"
COMPARISON = ROOT / "data" / "evaluation" / "intent_suggestions_before_after_comparison.csv"
INTENTS = {
    "APP_TECHNICAL_COMPATIBILITY", "SUBSCRIPTION_PLAN", "BILLING_PAYMENT_REFUND",
    "PLAYLIST_LIBRARY_DISCOVERY", "DOWNLOADS_OFFLINE_STORAGE", "ACCOUNT_ACCESS_SECURITY",
    "PLAYBACK_AUDIO", "COMMUNITY_INTEGRATIONS", "CONTENT_AVAILABILITY_METADATA",
    "FEATURE_REQUEST_PRODUCT_FEEDBACK", "OTHER_UNCLEAR",
}

def main():
    suggestions = pd.read_csv(SUGGESTIONS, keep_default_na=False)
    queue = pd.read_csv(QUEUE, keep_default_na=False)
    assert len(suggestions) == 200, f"Expected 200 examples, found {len(suggestions)}"
    assert len(suggestions["example_id"].unique()) == 200, "Duplicate example IDs found"
    assert set(suggestions["suggested_intent"]).issubset(INTENTS), "Suggestion outside taxonomy"
    assert set(suggestions["confidence"]).issubset({"High", "Medium", "Low"}), "Invalid confidence"
    assert suggestions["confidence_score"].equals(suggestions["confidence"]), "Confidence fields disagree"
    assert suggestions["requires_human_review"].astype(str).str.upper().isin({"TRUE", "FALSE"}).all(), "Invalid review flag"
    flagged_ids = set(suggestions.loc[suggestions["requires_human_review"], "example_id"])
    queue_ids = set(queue["example_id"])
    assert flagged_ids == queue_ids, "Review queue does not exactly match flagged suggestions"
    assert len(queue) == len(flagged_ids), "Duplicate or missing rows in review queue"
    assert list(suggestions.columns) == list(queue.columns), "Queue schema differs from suggestions schema"
    for path in (SUGGESTIONS, QUEUE):
        pd.read_csv(path, keep_default_na=False)

    before = pd.read_csv(BEFORE, keep_default_na=False) if BEFORE.exists() else None
    if before is not None:
        assert len(before) == 200 and set(before["example_id"]) == set(suggestions["example_id"]), "Invalid before snapshot"
        comparison = pd.DataFrame({
            "example_id": suggestions["example_id"],
            "before_intent": before["suggested_intent"],
            "after_intent": suggestions["suggested_intent"],
            "before_confidence": before["confidence"],
            "after_confidence": suggestions["confidence"],
            "before_flagged": before["requires_human_review"],
            "after_flagged": suggestions["requires_human_review"],
            "after_reason": suggestions["confidence_reason"],
        })
        comparison["moved_from_other_unclear"] = (comparison["before_intent"] == "OTHER_UNCLEAR") & (comparison["after_intent"] != "OTHER_UNCLEAR")
        comparison.to_csv(COMPARISON, index=False)

    approved = overridden = 0
    if PROGRESS.exists():
        progress = pd.read_csv(PROGRESS, keep_default_na=False)
        assert len(progress) == 200, "Progress file must retain all 200 examples for resume"
        approved = int((progress.get("label_action", pd.Series(dtype=str)) == "human_approved_suggestion").sum())
        overridden = int((progress.get("label_action", pd.Series(dtype=str)) == "human_override").sum())

    print("Validation: PASS")
    print(f"Total examples: {len(suggestions)}")
    print(f"High confidence count: {int((suggestions['confidence'] == 'High').sum())}")
    print(f"Medium confidence count: {int((suggestions['confidence'] == 'Medium').sum())}")
    print(f"Low confidence count: {int((suggestions['confidence'] == 'Low').sum())}")
    print(f"Flagged for human review count: {len(queue)}")
    print(f"OTHER_UNCLEAR count: {int((suggestions['suggested_intent'] == 'OTHER_UNCLEAR').sum())}")
    print(f"Human-approved labels: {approved}")
    print(f"Human-overridden labels: {overridden}")
    print("\nCount by suggested intent:")
    print(suggestions["suggested_intent"].value_counts().reindex(sorted(INTENTS), fill_value=0).to_string())
    if before is not None:
        def metrics(frame):
            return {
                "OTHER_UNCLEAR": int((frame["suggested_intent"] == "OTHER_UNCLEAR").sum()),
                "High": int((frame["confidence"] == "High").sum()),
                "Medium": int((frame["confidence"] == "Medium").sum()),
                "Low": int((frame["confidence"] == "Low").sum()),
                "Flagged": int(frame["requires_human_review"].sum()),
            }
        print("\nBefore vs after:")
        print(f"{'Metric':<22} {'Before':>8} {'After':>8} {'Change':>8}")
        bm, am = metrics(before), metrics(suggestions)
        for name in ("OTHER_UNCLEAR", "High", "Medium", "Low", "Flagged"):
            print(f"{name:<22} {bm[name]:>8} {am[name]:>8} {am[name]-bm[name]:>+8}")
        moved = comparison[comparison["moved_from_other_unclear"]]
        print(f"\nExamples moved from OTHER_UNCLEAR to a specific intent: {len(moved)}")
        for row in moved.head(10).itertuples(index=False):
            print(f"- {row.example_id}: {row.before_intent} -> {row.after_intent} ({row.after_confidence}); {row.after_reason}")
        print(f"Detailed comparison: {COMPARISON}")

if __name__ == "__main__":
    main()
