"""Generate deterministic, retrieval-grounded Spotify support drafts.

This module deliberately does not call an LLM or copy a historical reply. It
extracts a small set of safe, supported next-step patterns from retrieved
replies and composes a new concise draft around them.
"""

import argparse
import json
from pathlib import Path
import re
import sys


ROOT_DIR = Path(__file__).resolve().parents[2]
RETRIEVAL_DIR = ROOT_DIR / "src" / "retrieval"
if str(RETRIEVAL_DIR) not in sys.path:
    sys.path.insert(0, str(RETRIEVAL_DIR))

from retrieve_similar import DEFAULT_DATASET, retrieve_similar  # noqa: E402


MIN_SIMILARITY = 0.15


def sanitize_historical_reply(reply):
    """Remove contact details, links, greetings, and case-specific fragments."""
    text = str(reply or "").strip()
    text = re.sub(r"https?://\S+|www\.\S+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[redacted contact detail]", text)
    text = re.sub(r"@[A-Za-z0-9_]+", "[redacted handle]", text)
    text = re.sub(r"^(?:hey|hi|hello)\s+(?!there\b)[^,!:.]{1,40}[,!:.]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:your|the user's?)\s+(?:account['’]s?\s+)?(?:username|email address|email|user id)(?:\s+and\s+(?:your\s+)?(?:username|email address|email|user id))?\b", "account details", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:DM|direct message)\s+(?:us|me)\b", "contact support", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*/[A-Z]{1,3}$", "", text)
    text = " ".join(
        sentence for sentence in re.split(r"(?<=[.!?])\s+", text)
        if not re.search(r"contact support|account details|username|email address|\bemail\b", sentence, re.IGNORECASE)
    )
    text = re.sub(r"\s+", " ", text).strip(" ,:-")
    return text


def _supported_actions(reply):
    """Extract safe next steps that are explicitly supported by one reply."""
    text = str(reply or "").lower()
    actions = []
    if re.search(r"(?:spotify|app|application).{0,45}version|version.{0,45}(?:spotify|app|application)", text):
        actions.append("share the Spotify app version")
    if re.search(r"operating system|os version|(?:android|ios).{0,30}version|version.{0,30}(?:android|ios)|device.{0,20}version|version.{0,20}device", text):
        actions.append("share the operating-system version")
    if re.search(r"what device|which device|device (?:are|is) (?:you|it) using|make and model|device details", text):
        actions.append("share the device details")
    if re.search(r"different device|another device|other device|other devices", text):
        actions.append("try a different device")
    if re.search(r"wi[- ]?fi|3g/4g|mobile data|(?:wi[- ]?fi|3g/4g|mobile data).{0,30}both|both.{0,30}(?:wi[- ]?fi|3g/4g|mobile data)", text):
        actions.append("check whether it happens on Wi-Fi, mobile data, or both")
    if re.search(r"network|internet connection|connectivity|check.{0,25}connection|connection.{0,25}(?:issue|problem|check)", text):
        actions.append("check the network connection")
    if re.search(r"restart|restarting|reboot", text):
        actions.append("restart the device")
    if re.search(r"uninstall(?:ing)? and reinstall|reinstall(?:ing)?", text):
        actions.append("try reinstalling the app")
    if re.search(r"screenshot", text):
        actions.append("send a screenshot of the error")
    if re.search(r"exact error|specific error|error message|what error", text):
        actions.append("share the exact error message")
    if re.search(r"what(?:'|’)s happening|what is happening|what happens|happening exactly|what exactly", text):
        actions.append("describe exactly what happens when the issue occurs")
    if re.search(r"steps to reproduce|reproduce the issue|exact steps", text):
        actions.append("describe the steps to reproduce the issue")
    if re.search(r"what have you tried|what steps have you tried|troubleshooting.*tried|already tried", text):
        actions.append("tell us what troubleshooting you have already tried")
    if re.search(r"country your account|account.*country|country.*account", text):
        actions.append("confirm the country set on the account")
    if re.search(r"sign out everywhere|log(?:ging)? out", text):
        actions.append("sign out and sign back in")
    if re.search(r"change (?:your )?password|password again", text):
        actions.append("change the account password")
    if re.search(r"incognito|private browser", text):
        actions.append("try the action in a private or incognito browser window")
    elif re.search(r"another browser|different browser|web browser", text):
        actions.append("try another browser")
    if re.search(r"downloads? unexpectedly removed|offline", text):
        actions.append("check the support steps for downloads or offline tracks being removed")
    return list(dict.fromkeys(actions))


def _evidence(results):
    evidence = []
    for result in results:
        evidence.append({
            "source_conversation_id": result.get("source_conversation_id"),
            "similarity_score": result["similarity_score"],
            "sanitized_historical_reply": sanitize_historical_reply(result["spotify_reply"]),
            "used_for_generation": False,
        })
    return evidence


def _fallback(message, evidence, reason):
    return {
        "customer_message": message,
        "generated_reply": "Thanks for reaching out. We need a little more information to safely suggest the right next step. Please share what device you’re using and any exact error message you see.",
        "retrieval_evidence": evidence,
        "generation_method": "deterministic_retrieval_grounded_fallback",
        "fallback_reason": reason,
    }


def generate_reply(message, top_k=3, dataset_path=DEFAULT_DATASET):
    """Retrieve evidence and compose a new grounded support draft."""
    query = str(message or "").strip()
    if not query:
        raise ValueError("The customer message cannot be empty.")

    try:
        results = retrieve_similar(query, top_k=top_k, dataset_path=dataset_path)
    except (FileNotFoundError, ValueError) as error:
        return _fallback(query, [], f"retrieval_failed: {error}")

    evidence = _evidence(results)
    relevant = [result for result in results if result["similarity_score"] >= MIN_SIMILARITY]
    if not relevant:
        return _fallback(query, evidence, f"no_result_reached_similarity_threshold_{MIN_SIMILARITY}")

    ranked_actions = []
    for result in relevant:
        actions = _supported_actions(result["spotify_reply"])
        if actions:
            ranked_actions.append((result["similarity_score"], actions, result))
    if not ranked_actions:
        return _fallback(query, evidence, "relevant_replies_contained_no_safe_supported_next_step")

    ranked_actions.sort(key=lambda item: -item[0])
    best_score, actions, best_result = ranked_actions[0]
    best_index = results.index(best_result)
    evidence[best_index]["used_for_generation"] = True
    if len(actions) == 1:
        action_text = actions[0]
    else:
        action_text = "; ".join(actions[:-1]) + ", and " + actions[-1]
    generated = (
        "Thanks for reaching out. Based on similar Spotify support cases, "
        f"the suggested next step is to {action_text}. "
        "If the issue continues, let us know what happens and we can look into it further."
    )
    return {
        "customer_message": query,
        "generated_reply": generated,
        "retrieval_evidence": evidence,
        "generation_method": "deterministic_retrieval_grounded_action_extraction",
        "grounding_similarity_score": best_score,
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Generate a retrieval-grounded Spotify support reply draft.")
    parser.add_argument("--message", required=True, help="New customer message.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of historical conversations to retrieve.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Optional retrieval corpus path.")
    args = parser.parse_args()
    if not str(args.message).strip():
        parser.error("The customer message cannot be empty.")
    if args.top_k <= 0:
        parser.error("top-k must be a positive integer.")
    result = generate_reply(args.message, args.top_k, args.dataset)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
