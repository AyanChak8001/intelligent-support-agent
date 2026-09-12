"""Deterministic escalation decision for the current support pipeline.

This module is an adapter over the existing runtime APIs:
* ``suggest_intents.classify`` supplies the intent/confidence signal.
* ``generate_reply.generate_reply`` supplies retrieval and grounding signals.

It does not modify or use the golden evaluation labels.
"""

import argparse
import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[2]
INTENT_DIR = ROOT_DIR / "src" / "intent_classification"
REPLY_DIR = ROOT_DIR / "src" / "reply_generation"
for directory in (INTENT_DIR, REPLY_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from suggest_intents import classify  # noqa: E402
from generate_reply import DEFAULT_DATASET, MIN_SIMILARITY, generate_reply  # noqa: E402


RISK_SENSITIVE_INTENTS = {"ACCOUNT_ACCESS_SECURITY"}


def decide_escalation(message, top_k=3, dataset_path=DEFAULT_DATASET):
    """Return a structured AUTO_HANDLE or ESCALATE decision."""
    customer_message = str(message or "").strip()
    if not customer_message:
        raise ValueError("The customer message cannot be empty.")
    if int(top_k) <= 0:
        raise ValueError("top-k must be a positive integer.")

    # The new message has no historical reply. The existing classifier accepts
    # a reply context, so the adapter passes an empty reply and keeps that
    # limitation visible in the returned signals.
    intent, confidence, intent_reason, requires_human_review = classify(customer_message, "")
    reply_result = generate_reply(customer_message, top_k=top_k, dataset_path=dataset_path)
    evidence = reply_result.get("retrieval_evidence", [])
    top_similarity = max((float(item.get("similarity_score", 0.0)) for item in evidence), default=0.0)
    grounding_score = reply_result.get("grounding_similarity_score")
    reply_fallback = reply_result.get("generation_method", "").endswith("fallback")
    weak_retrieval = not evidence or top_similarity < MIN_SIMILARITY

    reasons = []
    if intent in RISK_SENSITIVE_INTENTS:
        reasons.append("account/security issues require human handling")
    if intent == "OTHER_UNCLEAR":
        reasons.append("the intent is OTHER_UNCLEAR")
    # ``requires_human_review`` belongs to the dataset-labeling workflow. It
    # indicates that a suggestion should be checked by a reviewer; it is not,
    # by itself, a production customer-escalation signal.
    low_confidence_weak_evidence = confidence == "Low" and (
        weak_retrieval or grounding_score is None
    )
    if low_confidence_weak_evidence:
        reasons.append("intent confidence is Low and the supporting evidence is weak")
    if weak_retrieval:
        reasons.append(f"retrieval evidence is weak (top similarity {top_similarity:.3f})")

    action = "ESCALATE" if reasons else "AUTO_HANDLE"
    if not reasons:
        if reply_fallback:
            reasons.append("specific non-sensitive intent and usable retrieval support a safe information-gathering fallback")
        else:
            reasons.append("specific intent, sufficient confidence, and grounded retrieval evidence are available")

    return {
        "customer_message": customer_message,
        "intent": intent,
        "confidence": confidence,
        "action": action,
        "reason": "; ".join(reasons) + ".",
        "signals": {
            "intent_reason": intent_reason,
            "requires_human_review": requires_human_review,
            "label_review_flag": requires_human_review,
            "review_flag_used_for_escalation": False,
            "risk_sensitive_intent": intent in RISK_SENSITIVE_INTENTS,
            "retrieval_result_count": len(evidence),
            "top_retrieval_similarity": top_similarity,
            "grounding_similarity_score": grounding_score,
            "weak_retrieval": weak_retrieval,
            "reply_generation_fallback": reply_fallback,
            "low_confidence_weak_evidence": low_confidence_weak_evidence,
            "generation_method": reply_result.get("generation_method"),
        },
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Decide whether a Spotify support case can be auto-handled.")
    parser.add_argument("--message", required=True, help="New customer message.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of historical conversations to retrieve.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Optional retrieval corpus path.")
    args = parser.parse_args()
    if not str(args.message).strip():
        parser.error("The customer message cannot be empty.")
    if args.top_k <= 0:
        parser.error("top-k must be a positive integer.")
    try:
        decision = decide_escalation(args.message, args.top_k, args.dataset)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
