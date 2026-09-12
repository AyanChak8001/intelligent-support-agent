"""Evaluate legitimate classifiers against the untouched golden dataset."""

import json
from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "results" / "golden_evaluation_dataset.csv"
INTENTS = [
    "APP_TECHNICAL_COMPATIBILITY", "SUBSCRIPTION_PLAN", "BILLING_PAYMENT_REFUND",
    "PLAYLIST_LIBRARY_DISCOVERY", "DOWNLOADS_OFFLINE_STORAGE", "ACCOUNT_ACCESS_SECURITY",
    "PLAYBACK_AUDIO", "COMMUNITY_INTEGRATIONS", "CONTENT_AVAILABILITY_METADATA",
    "FEATURE_REQUEST_PRODUCT_FEEDBACK", "OTHER_UNCLEAR",
]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_rules import predict  # noqa: E402
from baseline_majority import assess_majority, descriptive_majority_intent  # noqa: E402
from train_classifier import assess_training_data  # noqa: E402


def metrics(name, truth, predictions):
    precision, recall, f1, support = precision_recall_fscore_support(truth, predictions, labels=INTENTS, zero_division=0)
    per_intent = [{"intent": intent, "precision": float(p), "recall": float(r), "f1": float(score), "support": int(s)} for intent, p, r, score, s in zip(INTENTS, precision, recall, f1, support)]
    matrix = confusion_matrix(truth, predictions, labels=INTENTS)
    return {"name": name, "available": True, "number_evaluated": len(truth), "accuracy": float(accuracy_score(truth, predictions)), "macro_f1": float(f1_score(truth, predictions, labels=INTENTS, average="macro", zero_division=0)), "weighted_f1": float(f1_score(truth, predictions, labels=INTENTS, average="weighted", zero_division=0)), "per_intent": per_intent, "confusion_matrix_labels": INTENTS, "confusion_matrix": matrix.tolist()}


def main():
    golden = pd.read_csv(GOLDEN, keep_default_na=False)
    if len(golden) != 200 or not golden["final_intent"].isin(INTENTS).all():
        raise ValueError("Golden evaluation dataset must contain exactly 200 valid final_intent labels.")
    predictions = [predict(row.customer_message, row.spotify_reply)["intent"] for row in golden.itertuples()]
    rule_metrics = metrics("rule_baseline", golden["final_intent"].tolist(), predictions)
    majority_intent, class_counts = descriptive_majority_intent(golden["final_intent"])
    majority_metrics = metrics("descriptive_majority_sanity_check", golden["final_intent"].tolist(), [majority_intent] * len(golden))
    majority_metrics.update({
        "independent_experiment": False,
        "label_leakage": True,
        "prediction_policy": "Predict the most frequent final_intent in this same 200-row evaluation dataset for every row.",
        "majority_intent": majority_intent,
        "class_counts_used_to_select_majority": class_counts,
        "methodological_note": "Descriptive class-distribution sanity check only; not a held-out or independently trained baseline.",
    })
    results = {"evaluation_dataset": str(GOLDEN), "data_leakage_note": "Golden final_intent labels are used only as evaluation truth for the rule baseline. The descriptive majority sanity check necessarily uses the same labels to select its constant class and is explicitly not an independent benchmark.", "classifiers": [rule_metrics], "baselines": [majority_metrics], "unavailable_approaches": {"majority_baseline_from_separate_labels": assess_majority(), "tfidf_logistic_regression": assess_training_data()}}
    out = ROOT / "results" / "intent_classification_evaluation.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    rule = results["classifiers"][0]
    pd.DataFrame(rule["per_intent"]).to_csv(ROOT / "results" / "rule_baseline_per_intent.csv", index=False)
    pd.DataFrame(rule["confusion_matrix"], index=INTENTS, columns=INTENTS).to_csv(ROOT / "results" / "rule_baseline_confusion_matrix.csv")
    majority_out = ROOT / "results" / "majority_baseline_evaluation.json"
    majority_out.write_text(json.dumps(majority_metrics, indent=2), encoding="utf-8")
    print(json.dumps({"rule_baseline": {k: rule_metrics[k] for k in ("number_evaluated", "accuracy", "macro_f1", "weighted_f1")}, "descriptive_majority_sanity_check": {k: majority_metrics[k] for k in ("number_evaluated", "accuracy", "macro_f1", "weighted_f1", "majority_intent")}, "separate_label_majority_available": results["unavailable_approaches"]["majority_baseline_from_separate_labels"]["available"], "tfidf_training_sufficient": results["unavailable_approaches"]["tfidf_logistic_regression"]["sufficient_separate_labeled_data"]}, indent=2))
    print(out)


if __name__ == "__main__":
    main()
