"""Deterministic evaluation harness for the completed support agent.

The golden dataset is read-only evaluation truth. No labels are passed to the
agent and no training is performed. End-to-end evaluation uses a fixed sample
and a temporary retrieval corpus with exact golden conversation IDs removed.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
GOLDEN_FILE = ROOT_DIR / "results" / "golden_evaluation_dataset.csv"
METADATA_FILE = ROOT_DIR / "results" / "golden_evaluation_metadata.json"
REVIEW_PROGRESS_FILE = ROOT_DIR / "data" / "evaluation" / "intent_suggestions_review_progress.csv"
DEFAULT_RETRIEVAL_DATASET = ROOT_DIR / "data" / "processed" / "spotify_customer_support_pairs_clean.csv"
INTENT_DIR = ROOT_DIR / "src" / "intent_classification"
AGENT_SRC = ROOT_DIR / "src"
for directory in (INTENT_DIR, AGENT_SRC):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from baseline_rules import predict as predict_rule  # noqa: E402
from evaluate_classifier import INTENTS, metrics  # noqa: E402
from support_agent import run_support_agent  # noqa: E402


DEFAULT_SAMPLE_SIZE = 20
DEFAULT_SEED = 42
MIN_SIMILARITY = 0.15
URL_OR_EMAIL = re.compile(r"https?://|www\.|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", re.IGNORECASE)
ACCOUNT_IDENTIFIER = re.compile(r"\b(?:username|user\s*id|account\s*id|customer\s*id|email\s+address)\b", re.IGNORECASE)
CORPUS_CUSTOMER_COLUMN = "customer_message_clean"
CORPUS_REPLY_COLUMN = "spotify_reply_clean"


def file_sha256(path):
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_conversation_id(value):
    try:
        return f"spotify-{int(float(value))}"
    except (TypeError, ValueError):
        return str(value).strip()


def prepare_retrieval_corpus(golden, dataset_path, temporary_directory):
    """Measure overlap and remove exact golden IDs from a temporary corpus."""
    info = {
        "dataset": str(dataset_path),
        "source_rows": None,
        "valid_rows": None,
        "golden_id_overlap_rows": 0,
        "golden_id_overlap_unique_ids": 0,
        "exact_customer_message_overlap_rows": 0,
        "exact_self_retrieval_prevented": False,
        "filtered_rows_removed": 0,
        "error": None,
    }
    try:
        corpus = pd.read_csv(dataset_path, keep_default_na=False)
    except Exception as error:
        info["error"] = f"Could not inspect retrieval corpus: {type(error).__name__}: {error}"
        return Path(dataset_path), info
    info["source_rows"] = len(corpus)
    missing = {CORPUS_CUSTOMER_COLUMN, CORPUS_REPLY_COLUMN}.difference(corpus.columns)
    if missing:
        info["error"] = f"Retrieval corpus missing required columns: {sorted(missing)}"
        return Path(dataset_path), info
    valid = corpus[
        corpus[CORPUS_CUSTOMER_COLUMN].astype(str).str.strip().ne("")
        & corpus[CORPUS_REPLY_COLUMN].astype(str).str.strip().ne("")
    ].copy()
    info["valid_rows"] = len(valid)
    golden_ids = set(golden["example_id"].astype(str))
    if "customer_tweet_id" in valid.columns:
        corpus_ids = valid["customer_tweet_id"].map(normalize_conversation_id)
        overlap_mask = corpus_ids.isin(golden_ids)
        info["golden_id_overlap_rows"] = int(overlap_mask.sum())
        info["golden_id_overlap_unique_ids"] = int(corpus_ids[overlap_mask].nunique())
    golden_messages = set(golden["customer_message"].astype(str).str.strip().str.lower())
    info["exact_customer_message_overlap_rows"] = int(valid[CORPUS_CUSTOMER_COLUMN].astype(str).str.strip().str.lower().isin(golden_messages).sum())
    if info["golden_id_overlap_rows"] and "customer_tweet_id" in valid.columns:
        filtered = valid.loc[~overlap_mask].copy()
        info["filtered_rows_removed"] = len(valid) - len(filtered)
        filtered_path = Path(temporary_directory) / "retrieval_corpus_without_golden_ids.csv"
        filtered.to_csv(filtered_path, index=False)
        info["exact_self_retrieval_prevented"] = True
        return filtered_path, info
    return Path(dataset_path), info


def evaluate_intent_classifier(golden):
    predictions = [predict_rule(row.customer_message, row.spotify_reply)["intent"] for row in golden.itertuples()]
    return metrics("rule_baseline", golden["final_intent"].tolist(), predictions)


def reply_checks(result):
    action = result.get("action")
    reply = result.get("generated_reply") or ""
    evidence = result.get("retrieval_evidence_summary") or []
    fallback = str(result.get("generation_method", "")).endswith("fallback")
    grounded = bool(
        action == "AUTO_HANDLE"
        and evidence
        and float(result.get("top_similarity_score") or 0.0) >= MIN_SIMILARITY
        and result.get("grounding_score") is not None
        and not fallback
    )
    return {
        "reply_nonempty_when_auto_handle": action != "AUTO_HANDLE" or bool(reply.strip()),
        "no_obvious_url_or_email_leak": not bool(URL_OR_EMAIL.search(reply)),
        "no_obvious_account_identifier_leak": not bool(ACCOUNT_IDENTIFIER.search(reply)),
        "basic_grounding_signal_present": grounded,
        "fallback_used": fallback,
        "no_customer_reply_when_escalated": action != "ESCALATE" or result.get("generated_reply") is None,
    }


def evaluate_end_to_end(golden, sample_size, seed, top_k, dataset_path):
    sample_size = min(int(sample_size), len(golden))
    sample = golden.sample(n=sample_size, random_state=seed).sort_values("example_id").reset_index(drop=True)
    records = []
    for row in sample.itertuples():
        result = run_support_agent(row.customer_message, top_k=top_k, dataset_path=dataset_path)
        checks = reply_checks(result)
        evidence = result.get("retrieval_evidence_summary") or []
        records.append({
            "example_id": row.example_id,
            "true_intent": row.final_intent,
            "customer_message": row.customer_message,
            "predicted_intent": result.get("predicted_intent"),
            "confidence": result.get("confidence"),
            "action": result.get("action"),
            "escalation_reason": result.get("escalation_reason"),
            "top_similarity_score": result.get("top_similarity_score"),
            "grounding_score": result.get("grounding_score"),
            "generation_method": result.get("generation_method"),
            "fallback_used": checks["fallback_used"],
            "generated_reply_present": bool(result.get("generated_reply")),
            "internal_draft_present": bool(result.get("internal_draft")),
            "retrieval_evidence_count": len(evidence),
            **checks,
        })
    return sample, pd.DataFrame(records)


def _distribution(records, intent_column):
    grouped = (
        records.assign(escalated=records["action"].eq("ESCALATE"))
        .groupby(intent_column, dropna=False)["escalated"]
        .agg(["count", "sum"])
        .rename(columns={"count": "cases", "sum": "escalated"})
        .reset_index()
    )
    grouped["auto_handle"] = grouped["cases"] - grouped["escalated"]
    grouped["escalation_percentage"] = (grouped["escalated"] / grouped["cases"] * 100).round(2)
    return grouped


def build_report(golden, intent_metrics, sample, records, seed, top_k, dataset_path, overlap_info, protected_hashes):
    total = len(records)
    action_counts = records["action"].value_counts().to_dict()
    auto_count = int(action_counts.get("AUTO_HANDLE", 0))
    escalate_count = int(action_counts.get("ESCALATE", 0))
    reason_counts = Counter(records.loc[records["action"] == "ESCALATE", "escalation_reason"].astype(str))
    check_columns = [
        "reply_nonempty_when_auto_handle", "no_obvious_url_or_email_leak",
        "no_obvious_account_identifier_leak", "basic_grounding_signal_present",
        "no_customer_reply_when_escalated",
    ]
    checks = {column: {"passed": int(records[column].sum()), "total": total} for column in check_columns}
    by_predicted_intent = _distribution(records, "predicted_intent")
    by_true_intent = _distribution(records, "true_intent")
    return {
        "evaluation_dataset": str(GOLDEN_FILE),
        "intent_classification": intent_metrics,
        "end_to_end": {
            "sample_size": total,
            "sample_seed": seed,
            "top_k": top_k,
            "sample_example_ids": sample["example_id"].tolist(),
            "reference_labels_used_only_for_evaluation": True,
            "predicted_intent_accuracy_on_sample": float((records["predicted_intent"] == records["true_intent"]).mean()),
            "action_counts": {str(key): int(value) for key, value in action_counts.items()},
            "auto_handle_count": auto_count,
            "escalate_count": escalate_count,
            "auto_handle_percentage": round(auto_count / total * 100, 2) if total else 0.0,
            "escalate_percentage": round(escalate_count / total * 100, 2) if total else 0.0,
            "fallback_count": int(records["fallback_used"].sum()),
            "reply_quality_checks": checks,
            "reply_quality_check_note": "These are deterministic automated checks, not human judgments of reply quality.",
            "escalation_reason_counts": dict(reason_counts),
            "escalation_by_predicted_intent": by_predicted_intent.to_dict(orient="records"),
            "escalation_by_true_intent": by_true_intent.to_dict(orient="records"),
        },
        "overlap_analysis": overlap_info,
        "protected_artifact_hashes": protected_hashes,
        "reproducibility": {
            "fixed_seed": seed,
            "sample_order_is_sorted_by_example_id": True,
            "outputs_have_no_timestamp_fields": True,
            "rerun_with_same_seed_should_match": True,
        },
        "safety_and_limitations": [
            "The golden final_intent labels are never passed to the support agent and are used only as evaluation truth.",
            "The end-to-end sample is deterministic; full intent metrics cover all 200 examples.",
            "Exact golden conversation IDs are removed from the temporary retrieval corpus where IDs are available, but exact customer-message overlap and broader source overlap may remain.",
            "The rule baseline was developed using prior inspection of golden examples, so its metrics may be optimistic rather than a blind benchmark.",
            "Automated reply checks do not establish human helpfulness, correctness, tone, or agreement.",
            "No LLM-as-a-judge, human reply evaluation, or human-versus-LLM agreement study was run.",
        ],
        "retrieval_dataset": str(dataset_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate intent classification and the deterministic support agent.")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="End-to-end sample size (default: 20).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic sample seed (default: 42).")
    parser.add_argument("--top-k", type=int, default=3, help="Retrieval top-k for each agent case.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_RETRIEVAL_DATASET, help="Retrieval corpus override.")
    args = parser.parse_args()
    if args.sample_size <= 0 or args.top_k <= 0:
        parser.error("sample-size and top-k must be positive integers.")
    if not GOLDEN_FILE.exists():
        parser.error(f"Golden dataset not found: {GOLDEN_FILE}")
    golden = pd.read_csv(GOLDEN_FILE, keep_default_na=False)
    if len(golden) != 200 or not golden["final_intent"].isin(INTENTS).all():
        parser.error("Golden evaluation dataset must contain exactly 200 valid final_intent labels.")

    protected_paths = {
        "golden_dataset": GOLDEN_FILE,
        "golden_metadata": METADATA_FILE,
        "review_progress": REVIEW_PROGRESS_FILE,
    }
    hashes_before = {name: file_sha256(path) for name, path in protected_paths.items()}
    intent_metrics = evaluate_intent_classifier(golden)
    with tempfile.TemporaryDirectory(prefix="hiver_eval_") as temporary_directory:
        retrieval_path, overlap_info = prepare_retrieval_corpus(golden, args.dataset, temporary_directory)
        sample, records = evaluate_end_to_end(golden, args.sample_size, args.seed, args.top_k, retrieval_path)
    hashes_after = {name: file_sha256(path) for name, path in protected_paths.items()}
    protected_hashes = {
        name: {"before": hashes_before[name], "after": hashes_after[name], "unchanged": hashes_before[name] == hashes_after[name]}
        for name in protected_paths
    }
    if not all(item["unchanged"] for item in protected_hashes.values()):
        raise RuntimeError("Protected golden or human-review artifact changed during evaluation.")

    report = build_report(golden, intent_metrics, sample, records, args.seed, args.top_k, args.dataset, overlap_info, protected_hashes)
    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    records.to_csv(results_dir / "end_to_end_agent_evaluation.csv", index=False)
    records[["example_id", "true_intent", "predicted_intent", "action", "escalation_reason"]].to_csv(results_dir / "escalation_cases.csv", index=False)
    records[["example_id", "action", "reply_nonempty_when_auto_handle", "no_obvious_url_or_email_leak", "no_obvious_account_identifier_leak", "basic_grounding_signal_present", "fallback_used", "no_customer_reply_when_escalated"]].to_csv(results_dir / "reply_quality_checks.csv", index=False)
    pd.concat([
        pd.DataFrame(report["end_to_end"]["escalation_by_predicted_intent"]).assign(distribution="predicted_intent"),
        pd.DataFrame(report["end_to_end"]["escalation_by_true_intent"]).assign(distribution="true_intent").rename(columns={"true_intent": "predicted_intent"}),
    ], ignore_index=True).to_csv(results_dir / "escalation_by_intent.csv", index=False)
    (results_dir / "support_agent_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "intent_accuracy": intent_metrics["accuracy"],
        "intent_macro_f1": intent_metrics["macro_f1"],
        "intent_weighted_f1": intent_metrics["weighted_f1"],
        "intent_examples": intent_metrics["number_evaluated"],
        "end_to_end_sample_size": len(records),
        "action_counts": report["end_to_end"]["action_counts"] if "action_counts" in report["end_to_end"] else {"AUTO_HANDLE": report["end_to_end"]["auto_handle_count"], "ESCALATE": report["end_to_end"]["escalate_count"]},
        "fallback_count": report["end_to_end"]["fallback_count"],
        "overlap": report["overlap_analysis"],
        "protected_artifacts_unchanged": all(item["unchanged"] for item in protected_hashes.values()),
    }, indent=2))
    print(f"Saved evaluation outputs under {results_dir}")


if __name__ == "__main__":
    main()
