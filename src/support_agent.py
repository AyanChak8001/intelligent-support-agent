"""End-to-end deterministic Spotify support-agent integration.

Pipeline: intent classification -> historical retrieval -> reply generation ->
escalation decision. Escalated drafts are explicitly kept out of the customer
response field.
"""

import argparse
import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
ESCALATION_DIR = ROOT_DIR / "src" / "escalation"
REPLY_DIR = ROOT_DIR / "src" / "reply_generation"
for directory in (ESCALATION_DIR, REPLY_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from decide_escalation import DEFAULT_DATASET, decide_escalation  # noqa: E402
from generate_reply import generate_reply  # noqa: E402


def _safe_failure(message, reason):
    return {
        "customer_message": str(message or "").strip(),
        "predicted_intent": "UNKNOWN",
        "confidence": "UNKNOWN",
        "intent_reason": "Intent classification was unavailable.",
        "action": "ESCALATE",
        "escalation_reason": reason,
        "generated_reply": None,
        "internal_draft": None,
        "retrieval_evidence_summary": [],
        "top_similarity_score": 0.0,
        "grounding_score": None,
        "generation_method": "unavailable_due_to_component_failure",
    }


def run_support_agent(message, top_k=3, dataset_path=DEFAULT_DATASET):
    """Run the existing pipeline and return one safe structured result."""
    customer_message = str(message or "").strip()
    if not customer_message:
        return _safe_failure(customer_message, "The customer message is empty; human review is required.")
    try:
        top_k = int(top_k)
        if top_k <= 0:
            return _safe_failure(customer_message, "top-k must be a positive integer; human review is required.")
    except (TypeError, ValueError):
        return _safe_failure(customer_message, "top-k is invalid; human review is required.")

    try:
        decision = decide_escalation(customer_message, top_k=top_k, dataset_path=dataset_path)
    except Exception as error:  # Fail closed for unexpected component failures.
        return _safe_failure(customer_message, f"Pipeline decision failed ({type(error).__name__}); human review is required.")

    try:
        reply_result = generate_reply(customer_message, top_k=top_k, dataset_path=dataset_path)
    except Exception as error:  # The customer must never receive an ungrounded draft.
        reply_result = None
        reply_error = f"Reply generation failed ({type(error).__name__}); human review is required."
    else:
        reply_error = None

    signals = decision.get("signals", {})
    evidence = (reply_result or {}).get("retrieval_evidence", [])
    evidence_summary = [
        {
            "source_conversation_id": item.get("source_conversation_id"),
            "similarity_score": item.get("similarity_score"),
            "used_for_generation": item.get("used_for_generation", False),
        }
        for item in evidence
    ]
    action = decision.get("action", "ESCALATE")
    escalation_reason = decision.get("reason", "No escalation reason was provided; human review is required.")
    if reply_error:
        action = "ESCALATE"
        escalation_reason = f"{escalation_reason} {reply_error}"

    draft = (reply_result or {}).get("generated_reply")
    generation_method = (reply_result or {}).get("generation_method") or signals.get("generation_method")
    return {
        "customer_message": customer_message,
        "predicted_intent": decision.get("intent", "UNKNOWN"),
        "confidence": decision.get("confidence", "UNKNOWN"),
        "intent_reason": signals.get("intent_reason", "Unavailable."),
        "action": action,
        "escalation_reason": escalation_reason,
        # Only AUTO_HANDLE exposes a draft as the customer response.
        "generated_reply": draft if action == "AUTO_HANDLE" else None,
        "internal_draft": draft if action == "ESCALATE" else None,
        "retrieval_evidence_summary": evidence_summary,
        "top_similarity_score": signals.get("top_retrieval_similarity", 0.0),
        "grounding_score": (reply_result or {}).get("grounding_similarity_score", signals.get("grounding_similarity_score")),
        "generation_method": generation_method or "unknown",
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Run the end-to-end Spotify support agent pipeline.")
    parser.add_argument("--message", nargs="?", const="", default="", help="New customer message.")
    parser.add_argument("--top-k", type=int, default=3, help="Historical conversations to retrieve.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Optional retrieval corpus path.")
    args = parser.parse_args()
    print(json.dumps(run_support_agent(args.message, args.top_k, args.dataset), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
