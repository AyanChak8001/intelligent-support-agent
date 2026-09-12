"""Explainable rule baseline using the fixed 11-intent taxonomy.

This baseline is not trained on the golden dataset. It reuses the deterministic
customer-message plus Spotify-reply scoring logic from suggest_intents.py.
"""

from pathlib import Path
import sys

try:
    from suggest_intents import INTENTS, classify
except ImportError:  # Supports importing this module from the repository root.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from suggest_intents import INTENTS, classify


def predict(customer_message, spotify_reply):
    """Return the rule baseline's intent prediction and explanation."""
    intent, confidence, reason, _ = classify(customer_message, spotify_reply)
    return {"intent": intent, "confidence": confidence, "reason": reason}


def predict_intent(customer_message, spotify_reply):
    return predict(customer_message, spotify_reply)["intent"]


if __name__ == "__main__":
    print("Rule baseline ready; run evaluate_classifier.py for evaluation.")
