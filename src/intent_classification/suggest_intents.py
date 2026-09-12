"""Generate conservative, reviewable intent suggestions for the fixed 200-row sample."""

from pathlib import Path
import re
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "intent_review_sample.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "intent_suggestions_all_200.csv"
QUEUE_PATH = PROJECT_ROOT / "data" / "evaluation" / "manual_review_queue.csv"
SAMPLE_SIZE = 200
INTENTS = [
    "APP_TECHNICAL_COMPATIBILITY", "SUBSCRIPTION_PLAN", "BILLING_PAYMENT_REFUND",
    "PLAYLIST_LIBRARY_DISCOVERY", "DOWNLOADS_OFFLINE_STORAGE", "ACCOUNT_ACCESS_SECURITY",
    "PLAYBACK_AUDIO", "COMMUNITY_INTEGRATIONS", "CONTENT_AVAILABILITY_METADATA",
    "FEATURE_REQUEST_PRODUCT_FEEDBACK", "OTHER_UNCLEAR",
]

# Weighted customer-message cues. Phrases are deliberately stronger than words.
CUSTOMER_RULES = {
    "ACCOUNT_ACCESS_SECURITY": [("log in", 5), ("login", 5), ("password", 5), ("hacked", 6), ("unauthorized", 6), ("someone.*account", 6), ("stolen", 5), ("sign in", 4), ("account.*blocked", 4), ("kick.*account", 5), ("access.*account", 4)],
    "BILLING_PAYMENT_REFUND": [("refund", 6), ("charged", 6), ("charge", 4), ("payment", 5), ("credit card", 5), ("debit card", 5), ("billing", 5), ("billed", 5), ("money back", 6), ("paid", 3), ("price", 3), ("purchase", 5), ("take my.*\$", 5), ("money", 4), ("pay", 3)],
    "SUBSCRIPTION_PLAN": [("premium", 4), ("subscription", 5), ("student", 4), ("family plan", 5), ("family account", 5), ("family.*add", 5), ("upgrade", 4), ("downgrade", 4), ("renew", 4), ("membership", 4), ("free account", 3), ("tariff", 4), ("afford.*month", 4), ("redeem code", 4)],
    "PLAYBACK_AUDIO": [("won't play", 6), ("wont play", 6), ("not playing", 6), ("playback", 5), ("paus", 5), ("stopped", 5), ("let me.*listen", 5), ("no sound", 6), ("audio", 4), ("sound", 3), ("beep", 5), ("crack", 4), ("64kbps", 5), ("streaming", 4), ("delay", 4), ("shuffle", 4), ("repeat", 4), ("skip", 3)],
    "DOWNLOADS_OFFLINE_STORAGE": [("offline", 6), ("download", 6), ("dled", 6), ("sd card", 6), ("storage", 5), ("space on", 4), ("saved.*offline", 5), ("delete.*download", 5), ("download.*removed", 6)],
    "APP_TECHNICAL_COMPATIBILITY": [("app", 2), ("android", 4), ("iphone", 4), ("ios", 4), ("phone", 3), ("desktop", 4), ("web player", 5), ("browser", 4), ("sign up", 4), ("something went wrong", 5), ("form", 3), ("look like this", 4), ("opening", 3), ("crash", 6), ("freez", 6), ("update", 4), ("bug", 4), ("not working", 4), ("doesn't work", 4), ("unsupported", 5), ("compatib", 5), ("timeout", 4)],
    "CONTENT_AVAILABILITY_METADATA": [("not available", 6), ("unavailable", 6), ("missing song", 6), ("missing album", 6), ("where('d| is| has).*?(song|album|track|music)", 6), ("(song|album|artist|track).*?(wrong|not|missing)", 6), ("don't have.*(song|artist|album|music)", 6), ("(ghostface|ashanti|ronnie james dio|me gustas tu|twice|lemonade|lay's|taylor swift)", 5), ("not on", 5), ("album.*(up|available)", 5), ("lyrics", 5), ("wrong artist", 6), ("release date", 5), ("catalogue", 4), ("catalog", 4), ("album.*missing", 6), ("songs.*(spotify|app)", 4)],
    "PLAYLIST_LIBRARY_DISCOVERY": [("playlist", 6), ("queue", 5), ("library", 5), ("discover weekly", 6), ("daily mix", 6), ("recommend", 5), ("saved songs", 5), ("algorithm", 4), ("sort", 4), ("other songs", 4), ("10,000 songs", 5), ("delete and start over", 4)],
    "FEATURE_REQUEST_PRODUCT_FEEDBACK": [("please add", 6), ("add a", 4), ("add an", 4), ("would be nice", 5), ("feature request", 6), ("why remove.*feature", 6), ("consider adding", 6), ("wish you", 4), ("hope to see", 4), ("can you add", 6), ("bring back", 5), ("bring spotify to", 6), ("lossless", 5), ("hi-fi", 5), ("user id changed", 5), ("option to choose.*username", 5), ("music facts", 4)],
    "COMMUNITY_INTEGRATIONS": [("sonos", 6), ("roku", 6), ("facebook", 5), ("chromecast", 6), ("alexa", 6), ("echo", 5), ("google home", 6), ("shazam", 5), ("smart.*speaker", 5)],
}
REPLY_RULES = {
    "ACCOUNT_ACCESS_SECURITY": ["password", "account", "sign out everywhere", "hacked"],
    "BILLING_PAYMENT_REFUND": ["charged", "payment", "refund", "card", "billing", "price", "purchase", "money"],
    "SUBSCRIPTION_PLAN": ["premium", "student", "family", "subscription", "plan", "redeem", "offers"],
    "PLAYBACK_AUDIO": ["play", "pausing", "audio", "sound", "device", "streaming", "listening", "quality", "beeps", "song", "ads"],
    "DOWNLOADS_OFFLINE_STORAGE": ["download", "offline", "storage", "sd card", "removed", "steps under"],
    "APP_TECHNICAL_COMPATIBILITY": ["app", "device", "version", "crash", "developers", "update", "browser", "error", "restarting"],
    "CONTENT_AVAILABILITY_METADATA": ["track", "song", "album", "artist", "available", "lyrics", "content", "release date", "catalog", "country"],
    "PLAYLIST_LIBRARY_DISCOVERY": ["playlist", "queue", "recommend", "library", "community", "rest of the queue"],
    "FEATURE_REQUEST_PRODUCT_FEEDBACK": ["feedback", "idea", "feature", "suggest", "team", "lossless", "votes", "available in all regions"],
    "COMMUNITY_INTEGRATIONS": ["roku", "sonos", "facebook", "device", "app"],
}

def _text(value):
    return str(value or "").lower().replace("’", "'").strip()

def _score(text, rules):
    return sum(weight for pattern, weight in rules if re.search(pattern, text))

def classify(customer_message, spotify_reply):
    customer, reply = _text(customer_message), _text(spotify_reply)
    customer_scores = {intent: _score(customer, rules) for intent, rules in CUSTOMER_RULES.items()}
    reply_scores = {intent: sum(cue in reply for cue in cues) for intent, cues in REPLY_RULES.items()}
    combined = {intent: customer_scores[intent] + min(reply_scores[intent] * 2, 6) for intent in CUSTOMER_RULES}
    ranked = sorted(combined.items(), key=lambda item: item[1], reverse=True)
    best_intent, best_score = ranked[0]
    second_score = ranked[1][1]
    reply_ranked = sorted(reply_scores.items(), key=lambda item: item[1], reverse=True)
    if (not customer_scores[best_intent] or len(customer) < 18) and reply_ranked[0][1] >= 2:
        context_intent, context_score = reply_ranked[0]
        return context_intent, "Low", "Customer message lacks a clear standalone issue; historical Spotify reply supplies limited contextual evidence.", True
    if not customer_scores[best_intent] or len(customer) < 18:
        return "OTHER_UNCLEAR", "Low", "No clear standalone issue or taxonomy-specific evidence in the customer message.", True
    corroborated = bool(reply_scores[best_intent])
    ambiguous = second_score > 0 and best_score - second_score <= 2
    inconsistent = (not corroborated and any(value >= 2 for intent, value in reply_scores.items() if intent != best_intent))
    if best_score >= 7 and not ambiguous and not inconsistent:
        confidence = "High" if corroborated or customer_scores[best_intent] >= 6 else "Medium"
    elif best_score >= 4:
        confidence = "Medium"
    else:
        confidence = "Low"
    reasons = [f"Customer evidence matched {best_intent.replace('_', ' ').lower()} cues."]
    if corroborated:
        reasons.append("Historical Spotify reply provides corroborating context.")
    if ambiguous:
        reasons.append("A second taxonomy intent is also plausible.")
    if inconsistent:
        reasons.append("Customer message and historical reply point to different issues.")
    if confidence != "High":
        reasons.append("Human review is required because the evidence is not sufficiently decisive.")
    return best_intent, confidence, " ".join(reasons), confidence != "High" or ambiguous or inconsistent

def generate():
    source = pd.read_csv(INPUT_PATH, keep_default_na=False)
    if len(source) != SAMPLE_SIZE:
        raise ValueError(f"Expected exactly {SAMPLE_SIZE} examples, found {len(source)}.")
    if source["customer_tweet_id"].duplicated().any():
        raise ValueError("The fixed sample contains duplicate customer_tweet_id values.")
    output = pd.DataFrame({
        "example_id": [f"spotify-{int(float(value))}" for value in source["customer_tweet_id"]],
        "customer_message": source["customer_message_clean"],
        "spotify_reply": source["spotify_reply_clean"],
    })
    results = [classify(row.customer_message, row.spotify_reply) for row in output.itertuples()]
    output[["suggested_intent", "confidence", "confidence_reason", "requires_human_review"]] = results
    output["confidence_score"] = output["confidence"]
    output["suggestion_source"] = "deterministic_heuristic_customer_and_reply"
    output["requires_human_review"] = output["requires_human_review"].astype(bool)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)
    output[output["requires_human_review"]].to_csv(QUEUE_PATH, index=False)
    return output

def main():
    output = generate()
    print(f"Generated {len(output)} automatic suggestions: {OUTPUT_PATH}")
    print(f"Flagged {int(output['requires_human_review'].sum())} for human review: {QUEUE_PATH}")
    print(output["confidence"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0).to_string())
    print("These are automatic suggestions, not human labels; no golden dataset was created.")

if __name__ == "__main__":
    main()
