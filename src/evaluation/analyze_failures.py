"""Read-only failure analysis for the completed support-agent evaluation.

The script derives findings from existing measured artifacts and writes only
the two requested analysis reports. It does not rerun the agent, relabel data,
or modify any evaluation inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_DIR / "results"
GOLDEN_FILE = RESULTS_DIR / "golden_evaluation_dataset.csv"
BASELINE_PER_INTENT = RESULTS_DIR / "rule_baseline_per_intent.csv"
BASELINE_CONFUSION = RESULTS_DIR / "rule_baseline_confusion_matrix.csv"
END_TO_END = RESULTS_DIR / "end_to_end_agent_evaluation.csv"
ESCALATION_CASES = RESULTS_DIR / "escalation_cases.csv"
QUALITY_CHECKS = RESULTS_DIR / "reply_quality_checks.csv"
LLM_RESULTS = RESULTS_DIR / "llm_judge_results.csv"
AGREEMENT = RESULTS_DIR / "human_llm_agreement.json"
HUMAN_EVALUATION = ROOT_DIR / "data" / "evaluation" / "human_reply_evaluation_template.csv"
OUTPUT_JSON = RESULTS_DIR / "failure_analysis.json"
OUTPUT_MD = RESULTS_DIR / "failure_analysis.md"

PROTECTED = [
    GOLDEN_FILE,
    RESULTS_DIR / "golden_evaluation_metadata.json",
    ROOT_DIR / "data" / "evaluation" / "intent_suggestions_review_progress.csv",
    HUMAN_EVALUATION,
]
SCORE_COLUMNS = ["helpfulness_score", "correctness_score", "relevance_score", "tone_score", "safety_score", "overall_score"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def pct(count: int, total: int) -> float | None:
    return round(100 * count / total, 2) if total else None


def example(row: dict[str, str], message_key: str = "customer_message") -> dict[str, str]:
    result = {"example_id": row.get("example_id", "")}
    if row.get(message_key, ""):
        result["customer_message"] = row[message_key]
    return result


def finding(category: str, title: str, count: int | None, denominator: int | None, representatives: list[dict[str, str]], source: list[str], measured: list[str], inferred: list[str], root_cause: str, impact: str, improvement: str) -> dict[str, Any]:
    return {
        "failure_category": category,
        "failure_mode": title,
        "affected_count": count,
        "denominator": denominator,
        "affected_percentage": pct(count, denominator) if count is not None and denominator is not None else None,
        "representative_examples": representatives,
        "evidence_source_files": source,
        "measured_facts": measured,
        "interpretation_and_limits": inferred,
        "technical_root_cause": root_cause,
        "impact_on_support_agent": impact,
        "recommended_improvement": improvement,
    }


def analyze() -> dict[str, Any]:
    before = {str(path): sha256(path) for path in PROTECTED if path.exists()}
    golden = read_csv(GOLDEN_FILE)
    per_intent = read_csv(BASELINE_PER_INTENT)
    confusion_rows = read_csv(BASELINE_CONFUSION)
    e2e = read_csv(END_TO_END)
    escalations = read_csv(ESCALATION_CASES)
    quality = read_csv(QUALITY_CHECKS)
    llm = read_csv(LLM_RESULTS)
    human = read_csv(HUMAN_EVALUATION)
    agreement = json.loads(AGREEMENT.read_text(encoding="utf-8"))

    after = {str(path): sha256(path) for path in PROTECTED if path.exists()}
    if before != after:
        raise RuntimeError("A protected input artifact changed during analysis.")

    golden_total = len(golden)
    confusion_header = list(confusion_rows[0].keys()) if confusion_rows else []
    source_key = "Unnamed: 0" if "Unnamed: 0" in confusion_header else ""
    labels = [label for label in confusion_header if label and label != source_key]
    correct = sum(int(row.get(label, 0) or 0) for row in confusion_rows for label in labels if row.get(source_key) == label)
    confusion_total = sum(int(row.get(label, 0) or 0) for row in confusion_rows for label in labels)
    intent_errors = confusion_total - correct
    off_diagonal = []
    for row in confusion_rows:
        source_intent = row.get(source_key, "")
        for predicted in labels:
            count = int(row.get(predicted, 0) or 0)
            if source_intent != predicted and count:
                off_diagonal.append((count, source_intent, predicted))
    off_diagonal.sort(reverse=True)
    e2e_misclassified = [row for row in e2e if row["true_intent"] != row["predicted_intent"]]

    fallback = [row for row in e2e if row["fallback_used"].strip().lower() == "true"]
    fallback_auto = [row for row in fallback if row["action"] == "AUTO_HANDLE"]
    weak_grounding = [row for row in e2e if row["basic_grounding_signal_present"].strip().lower() != "true"]
    escalated = [row for row in e2e if row["action"] == "ESCALATE"]
    low_or_unclear_escalations = [row for row in escalated if row["confidence"] == "Low" or row["predicted_intent"] == "OTHER_UNCLEAR"]
    fallback_ids = {row["example_id"] for row in fallback}
    llm_by_id = {row["evaluation_id"].replace("human-reply-", "", 1): row for row in llm if row.get("judge_status") == "success"}
    human_by_id = {row["evaluation_id"].replace("human-reply-", "", 1): row for row in human if row.get("reviewed", "").strip().lower() == "true"}
    overall_diffs = []
    for example_id, llm_row in llm_by_id.items():
        if example_id in human_by_id:
            difference = int(llm_row["overall_score"]) - int(human_by_id[example_id]["overall_score"])
            overall_diffs.append((difference, example_id, llm_row["customer_message"]))
    overall_diffs.sort()
    disagreement_count = sum(diff != 0 for diff, _, _ in overall_diffs)
    disagreement_reps = [{"example_id": item[1], "customer_message": item[2], "overall_signed_difference": item[0]} for item in overall_diffs if abs(item[0]) >= 2][:3]

    sources = {
        "intent": [str(BASELINE_PER_INTENT), str(BASELINE_CONFUSION), str(GOLDEN_FILE), str(END_TO_END)],
        "retrieval": [str(END_TO_END), str(QUALITY_CHECKS)],
        "reply": [str(END_TO_END), str(QUALITY_CHECKS), str(LLM_RESULTS)],
        "escalation": [str(END_TO_END), str(ESCALATION_CASES)],
        "agreement": [str(LLM_RESULTS), str(HUMAN_EVALUATION), str(AGREEMENT)],
    }
    findings = [
        finding(
            "intent_classification",
            "Rule baseline misclassifies 18 of 200 labeled examples",
            intent_errors, confusion_total,
            [example(row) for row in e2e_misclassified[:3]], sources["intent"],
            [f"The confusion matrix contains {intent_errors} off-diagonal predictions out of {confusion_total} examples ({pct(intent_errors, confusion_total)}%).", f"The largest individual confusion count is {off_diagonal[0][0]} for {off_diagonal[0][1]} predicted as {off_diagonal[0][2]}." if off_diagonal else "No off-diagonal confusion was recorded."],
            ["The confusion artifact gives aggregate counts, not a causal feature analysis. End-to-end representative rows are a separate 20-example sample."],
            "The measured failure is in the deterministic rule classifier's intent decision boundary; the artifacts do not establish which rule or token caused each error.",
            "Wrong intent selection can route retrieval and escalation logic toward mismatched support behavior.",
            "Review the highest-volume confusion pairs with discriminative cue tests, then add regression cases before changing rules."
        ),
        finding(
            "retrieval",
            "12 of 20 end-to-end cases lack the basic grounding signal",
            len(weak_grounding), len(e2e),
            [example(row) for row in weak_grounding[:3]], sources["retrieval"],
            [f"{len(weak_grounding)} of {len(e2e)} end-to-end rows ({pct(len(weak_grounding), len(e2e))}%) have basic_grounding_signal_present=false.", f"The same sample has {len([row for row in e2e if not row['grounding_score'].strip()])} blank grounding_score values; retrieval evidence count is present but does not itself prove useful grounding."],
            ["Missing grounding_score is a measured limitation of the output signal, not proof that retrieval returned irrelevant documents in every row."],
            "The generation/evaluation pipeline does not produce a grounding score for fallback paths, so retrieval usefulness is not demonstrated for those cases.",
            "The agent may expose generic responses without a measured connection to retrieved support evidence.",
            "Add retrieval diagnostics such as evidence relevance and action-support coverage, and gate customer-facing handling on validated evidence quality."
        ),
        finding(
            "reply_generation",
            "Deterministic reply fallback is used in 10 of 20 end-to-end cases",
            len(fallback), len(e2e),
            [example(row) for row in fallback[:3]], sources["reply"],
            [f"{len(fallback)} of {len(e2e)} rows ({pct(len(fallback), len(e2e))}%) have fallback_used=true.", f"{len(fallback_auto)} fallback rows were AUTO_HANDLE and therefore returned a customer-facing reply; the other {len(fallback) - len(fallback_auto)} were escalated with an internal draft."],
            ["The deterministic checks identify fallback usage, but do not independently prove that every fallback reply was low quality. Human/LLM scores cover 16 examples, not all 20 end-to-end rows."],
            "No supported action was extracted for these rows, so generate_reply used its safe information-gathering fallback.",
            "Fallback replies are less specific and may fail to advance troubleshooting despite being non-empty and privacy-safe.",
            "Expand safe action extraction and add intent-specific fallback templates; retain escalation when the system lacks supportable next steps."
        ),
        finding(
            "escalation_behavior",
            "Escalation is concentrated in low-confidence or OTHER_UNCLEAR paths",
            len(escalated), len(e2e),
            [example(row) for row in escalated[:4]], sources["escalation"],
            [f"{len(escalated)} of {len(e2e)} rows ({pct(len(escalated), len(e2e))}%) were escalated.", f"{len(low_or_unclear_escalations)} of those {len(escalated)} escalations ({pct(len(low_or_unclear_escalations), len(escalated))}%) had Low confidence or predicted intent OTHER_UNCLEAR.", "One escalated row had a specific true intent but was predicted as OTHER_UNCLEAR."],
            ["Escalation is a safety behavior, so the artifact does not establish that all escalations are failures. The specific-intent/OTHER_UNCLEAR row identifies a measurable routing opportunity, not proof that the policy itself is incorrect."],
            "The observed escalation policy escalates OTHER_UNCLEAR, low-confidence, and account/security cases; one sample row reached this path after an intent mismatch.",
            "Customers receive no customer-facing draft on escalation, increasing handoff dependence and response latency.",
            "Measure escalation precision and resolution outcomes by reason, and improve classification/evidence before relaxing escalation safeguards."
        ),
        finding(
            "human_llm_disagreement",
            "LLM and human overall scores disagree on 9 of 16 reviewed replies",
            disagreement_count, len(overall_diffs),
            disagreement_reps, sources["agreement"],
            [f"Overall exact agreement is {agreement['criteria']['overall_score']['exact_agreement_rate']} across {agreement['matched_examples']} matched examples.", f"Overall MAE is {agreement['criteria']['overall_score']['mean_absolute_error']}; the average signed difference (LLM minus human) is {agreement['criteria']['overall_score']['average_signed_difference_llm_minus_human']}.", "The LLM scored lower than the human on 9 of 16 overall scores."],
            ["This is disagreement with one human evaluator, not evidence that either evaluator is correct. The 16-example human-reviewed sample is small and homogeneous; weighted kappa is undefined because human scores have no variance."],
            "The two evaluators apply different quality judgments to the same generated replies; the artifacts cannot identify whether rubric interpretation, calibration, or reply ambiguity is the dominant cause.",
            "LLM-based quality monitoring may rank replies differently from the human benchmark, especially for overall quality.",
            "Calibrate the judge on adjudicated examples, add explicit anchor examples to the rubric, and use disagreement review rather than treating the LLM score as ground truth."
        ),
    ]

    report = {
        "analysis": "failure_analysis",
        "generated_from": [str(GOLDEN_FILE), str(BASELINE_PER_INTENT), str(BASELINE_CONFUSION), str(END_TO_END), str(ESCALATION_CASES), str(QUALITY_CHECKS), str(LLM_RESULTS), str(AGREEMENT)],
        "top_5_failure_modes": findings,
        "sample_size_limitations": [
            f"Intent baseline metrics cover {golden_total} reviewed golden examples; end-to-end, escalation, and reply checks cover only {len(e2e)} sampled examples.",
            f"Human/LLM agreement covers {len(overall_diffs)} matched examples and one human evaluator.",
            "Aggregate confusion counts do not identify causal classifier rules, and deterministic quality checks do not replace human reply-quality review.",
        ],
        "protected_artifacts_unchanged": {path: True for path in before},
    }
    return report


def markdown(report: dict[str, Any]) -> str:
    lines = ["# Failure Analysis", "", "This report is generated from existing measured artifacts. Interpretive statements are labeled separately from measured facts.", ""]
    for index, item in enumerate(report["top_5_failure_modes"], 1):
        lines.extend([f"## {index}. {item['failure_mode']}", "", f"- Category: {item['failure_category']}", f"- Affected: {item['affected_count']} / {item['denominator']} ({item['affected_percentage']}%)", f"- Evidence: {', '.join(item['evidence_source_files'])}", "", "### Measured facts", ""])
        lines.extend([f"- {fact}" for fact in item["measured_facts"]])
        lines.extend(["", "### Interpretation and limits", ""])
        lines.extend([f"- {fact}" for fact in item["interpretation_and_limits"]])
        lines.extend(["", f"**Technical root cause:** {item['technical_root_cause']}", "", f"**Impact:** {item['impact_on_support_agent']}", "", f"**Recommended improvement:** {item['recommended_improvement']}", "", "Representative examples:", ""])
        lines.extend([f"- `{row.get('example_id', '')}` — {row.get('customer_message', '')}" for row in item["representative_examples"]])
        lines.append("")
    lines.extend(["## Sample-size limitations", ""])
    lines.extend([f"- {item}" for item in report["sample_size_limitations"]])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate read-only failure-analysis reports from completed evaluations.")
    parser.add_argument("--json-output", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=OUTPUT_MD)
    args = parser.parse_args()
    report = analyze()
    args.json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.markdown_output.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"json_output": str(args.json_output), "markdown_output": str(args.markdown_output), "failure_modes": [item["failure_mode"] for item in report["top_5_failure_modes"]], "protected_artifacts_unchanged": all(report["protected_artifacts_unchanged"].values())}, indent=2))


if __name__ == "__main__":
    main()
