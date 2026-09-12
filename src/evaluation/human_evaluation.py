"""Generate and validate the manual evaluation sheet for customer-facing replies.

This workflow never fills human scores. The generated sheet contains only the
deterministic AUTO_HANDLE cases from the recorded end-to-end evaluation.
"""

import argparse
import json
from pathlib import Path
import sys
import tempfile

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_DIR / "results"
E2E_RESULTS = RESULTS_DIR / "end_to_end_agent_evaluation.csv"
E2E_REPORT = RESULTS_DIR / "support_agent_evaluation.json"
GOLDEN_FILE = RESULTS_DIR / "golden_evaluation_dataset.csv"
DEFAULT_TEMPLATE = ROOT_DIR / "data" / "evaluation" / "human_reply_evaluation_template.csv"
DEFAULT_SUMMARY = RESULTS_DIR / "human_reply_evaluation_summary.json"

AGENT_SRC = ROOT_DIR / "src"
if str(AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(AGENT_SRC))

from evaluation.evaluate_system import DEFAULT_RETRIEVAL_DATASET, prepare_retrieval_corpus  # noqa: E402
from reply_generation.generate_reply import generate_reply  # noqa: E402
from support_agent import run_support_agent  # noqa: E402


SCORE_COLUMNS = [
    "helpfulness_score",
    "correctness_score",
    "relevance_score",
    "tone_score",
    "safety_score",
    "overall_score",
]
REQUIRED_COLUMNS = [
    "evaluation_id",
    "customer_message",
    "predicted_intent",
    "generated_reply",
    "retrieval_evidence_summary",
    *SCORE_COLUMNS,
    "evaluator_id",
    "evaluator_notes",
    "reviewed",
]


def _parse_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def generate_template(output_path=DEFAULT_TEMPLATE):
    """Reconstruct recorded AUTO_HANDLE outputs and leave all human fields blank."""
    if not E2E_RESULTS.exists() or not E2E_REPORT.exists():
        raise FileNotFoundError("Existing end-to-end evaluation outputs are required.")

    e2e = pd.read_csv(E2E_RESULTS, keep_default_na=False)
    missing = {"example_id", "customer_message", "predicted_intent", "action"}.difference(e2e.columns)
    if missing:
        raise ValueError(f"End-to-end results missing required columns: {sorted(missing)}")
    auto = e2e[e2e["action"].astype(str).eq("AUTO_HANDLE")].copy()

    report = json.loads(E2E_REPORT.read_text(encoding="utf-8"))
    seed = report["end_to_end"]["sample_seed"]
    top_k = report["end_to_end"]["top_k"]
    golden = pd.read_csv(GOLDEN_FILE, keep_default_na=False)

    rows = []
    # Filtering uses IDs only to prevent exact self-retrieval. Golden true
    # labels are never passed to the agent or written to the human worksheet.
    with tempfile.TemporaryDirectory(prefix="hiver_human_eval_") as temporary_directory:
        retrieval_path, overlap = prepare_retrieval_corpus(
            golden, DEFAULT_RETRIEVAL_DATASET, temporary_directory
        )
        for item in auto.itertuples(index=False):
            result = run_support_agent(item.customer_message, top_k=top_k, dataset_path=retrieval_path)
            if result.get("action") != "AUTO_HANDLE" or not str(result.get("generated_reply") or "").strip():
                raise RuntimeError(f"Recorded AUTO_HANDLE case did not reproduce safely: {item.example_id}")
            evidence_result = generate_reply(item.customer_message, top_k=top_k, dataset_path=retrieval_path)
            safe_evidence = [
                {
                    "source_conversation_id": evidence.get("source_conversation_id"),
                    "similarity_score": evidence.get("similarity_score"),
                    "sanitized_historical_reply": evidence.get("sanitized_historical_reply"),
                    "used_for_generation": evidence.get("used_for_generation", False),
                }
                for evidence in evidence_result.get("retrieval_evidence", [])
            ]
            rows.append({
                "evaluation_id": f"human-reply-{item.example_id}",
                "customer_message": item.customer_message,
                "predicted_intent": item.predicted_intent,
                "generated_reply": result["generated_reply"],
                "retrieval_evidence_summary": json.dumps(safe_evidence, ensure_ascii=False),
                **{column: "" for column in SCORE_COLUMNS},
                "evaluator_id": "",
                "evaluator_notes": "",
                "reviewed": "false",
            })

    output = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Generated {len(output)} manual evaluation rows: {output_path}")
    print(f"Source sample size: {report['end_to_end']['sample_size']}; seed: {seed}; top-k: {top_k}")
    print(f"Exact self-retrieval prevention used: {overlap['exact_self_retrieval_prevented']}")
    print("Human score columns were left blank.")


def validate_and_aggregate(input_path=DEFAULT_TEMPLATE, summary_path=DEFAULT_SUMMARY):
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Human evaluation file not found: {input_path}")
    data = pd.read_csv(input_path, keep_default_na=False)
    missing = set(REQUIRED_COLUMNS).difference(data.columns)
    if missing:
        raise ValueError(f"Human evaluation file missing required columns: {sorted(missing)}")

    invalid_scores = []
    incomplete = []
    completed_indices = []
    for index, row in data.iterrows():
        row_id = row["evaluation_id"]
        row_invalid = False
        for column in SCORE_COLUMNS:
            value = str(row[column]).strip()
            if value == "":
                continue
            try:
                number = int(value)
            except ValueError:
                invalid_scores.append({"evaluation_id": row_id, "field": column, "value": value})
                row_invalid = True
                continue
            if number not in range(1, 6):
                invalid_scores.append({"evaluation_id": row_id, "field": column, "value": value})
                row_invalid = True
        reviewed = _parse_bool(row["reviewed"])
        all_scores = all(str(row[column]).strip() != "" for column in SCORE_COLUMNS)
        if reviewed and (not all_scores or not str(row["evaluator_id"]).strip() or row_invalid):
            invalid_scores.append({"evaluation_id": row_id, "field": "reviewed", "value": "true without valid completed scores/evaluator"})
        if reviewed and all_scores and not row_invalid:
            completed_indices.append(index)
        else:
            incomplete.append(row_id)

    completed = data.loc[completed_indices]
    summary = {
        "source_file": str(input_path),
        "total_template_rows": int(len(data)),
        "completed_human_evaluations": int(len(completed)),
        "incomplete_evaluations": int(len(incomplete)),
        "incomplete_evaluation_ids": incomplete,
        "invalid_score_count": len(invalid_scores),
        "invalid_scores": invalid_scores,
        "score_scale": "1-5 integer scores; aggregates include only reviewed=true rows with all six scores valid",
    }
    if not completed.empty:
        numeric = completed[SCORE_COLUMNS].astype(int)
        summary["average_score_by_criterion"] = {
            column: float(numeric[column].mean()) for column in SCORE_COLUMNS
        }
        summary["score_distribution"] = {
            column: {str(score): int((numeric[column] == score).sum()) for score in range(1, 6)}
            for column in SCORE_COLUMNS
        }
        summary["overall_average"] = float(numeric["overall_score"].mean())
        summary["status"] = "completed_reviews_available"
    else:
        summary["status"] = "No completed human evaluations available"

    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if completed.empty:
        print("No completed human evaluations available")
    else:
        print(f"Completed human evaluations: {len(completed)}")
        print(json.dumps(summary, indent=2))
    print(f"Saved validation summary: {summary_path}")
    return summary


def _display_example(row, position, total):
    print("\n" + "=" * 50)
    print(f"EXAMPLE {position}/{total}")
    print("=" * 50)
    print("\nCUSTOMER MESSAGE:")
    print(row["customer_message"])
    print("\nPREDICTED INTENT:")
    print(row["predicted_intent"])
    print("\nGENERATED REPLY:")
    print(row["generated_reply"])
    print("\n" + "=" * 50)


def _ask_score(label):
    while True:
        value = input(f"{label} score (1-5): ").strip()
        if value in {"1", "2", "3", "4", "5"}:
            return int(value)
        print("Please enter an integer from 1 through 5.")


def _save_data_atomically(data, input_path):
    """Write the confirmed update through a same-directory temporary file."""
    input_path = Path(input_path)
    temporary_path = input_path.with_name(f".{input_path.name}.tmp")
    try:
        data.to_csv(temporary_path, index=False)
        temporary_path.replace(input_path)
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def interactive_review(input_path=DEFAULT_TEMPLATE):
    """Interactively score pending rows, saving only after explicit confirmation."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Human evaluation file not found: {input_path}")
    data = pd.read_csv(input_path, keep_default_na=False)
    missing = set(REQUIRED_COLUMNS).difference(data.columns)
    if missing:
        raise ValueError(f"Human evaluation file missing required columns: {sorted(missing)}")
    # Blank score columns may load as pandas StringDtype. Normalize only these
    # editable columns so Python integer scores can be assigned safely.
    for column in SCORE_COLUMNS:
        data[column] = data[column].astype(object)

    pending = [index for index, row in data.iterrows() if not _parse_bool(row["reviewed"])]
    total = len(data)
    if not pending:
        print("All human evaluation rows are already reviewed.")
        return

    print(f"Loaded {total} rows. Resuming at the first unreviewed row.")
    try:
        for pending_number, index in enumerate(pending, start=1):
            while True:
                row = data.loc[index]
                _display_example(row, index + 1, total)
                command = input("Enter=continue, r=redisplay, s=skip, q=quit: ").strip().lower()
                if command == "q":
                    print("Quit safely. No unsaved evaluation was changed.")
                    return
                if command == "r":
                    continue
                if command == "s":
                    print("Skipped without modifying this row.")
                    break
                if command != "":
                    print("Please press Enter, or choose r, s, or q.")
                    continue

                scores = {
                    "helpfulness_score": _ask_score("Helpfulness"),
                    "correctness_score": _ask_score("Correctness"),
                    "relevance_score": _ask_score("Relevance"),
                    "tone_score": _ask_score("Tone"),
                    "safety_score": _ask_score("Safety"),
                    "overall_score": _ask_score("Overall"),
                }
                evaluator_id = ""
                while not evaluator_id:
                    evaluator_id = input("evaluator_id: ").strip()
                    if not evaluator_id:
                        print("evaluator_id cannot be empty for a completed review.")
                notes = input("evaluation_notes (optional): ")

                print("\nEntered scores:")
                for column, score in scores.items():
                    print(f"  {column}: {score}")
                print(f"  evaluator_id: {evaluator_id}")
                print(f"  evaluation_notes: {notes}")
                confirmation = input("Save this evaluation? (y/n): ").strip().lower()
                if confirmation == "y":
                    for column, score in scores.items():
                        data.at[index, column] = score
                    data.at[index, "evaluator_id"] = str(evaluator_id)
                    data.at[index, "evaluator_notes"] = str(notes)
                    data.at[index, "reviewed"] = True
                    _save_data_atomically(data, input_path)
                    print(f"Saved evaluation {pending_number}/{len(pending)} immediately.")
                    break
                print("Evaluation not saved; the current example will be shown again.")
    except (KeyboardInterrupt, EOFError):
        print("\nQuit safely. No unsaved evaluation was changed.")


def main():
    parser = argparse.ArgumentParser(description="Create and validate human reply evaluations.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate-template")
    generate_parser.add_argument("--output", type=Path, default=DEFAULT_TEMPLATE)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--input", type=Path, default=DEFAULT_TEMPLATE)
    validate_parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    interactive_parser = subparsers.add_parser("interactive")
    interactive_parser.add_argument("--input", type=Path, default=DEFAULT_TEMPLATE)
    args = parser.parse_args()
    if args.command == "generate-template":
        generate_template(args.output)
    elif args.command == "validate":
        validate_and_aggregate(args.input, args.summary_output)
    else:
        interactive_review(args.input)


if __name__ == "__main__":
    main()
