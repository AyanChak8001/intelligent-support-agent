"""Retrieve similar historical Spotify support conversations with TF-IDF.

The retrieval corpus is the existing cleaned historical conversation file,
not the 200-example golden evaluation dataset:
data/processed/spotify_customer_support_pairs_clean.csv
"""

import argparse
from pathlib import Path
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT_DIR / "data" / "processed" / "spotify_customer_support_pairs_clean.csv"
CUSTOMER_COLUMN = "customer_message_clean"
REPLY_COLUMN = "spotify_reply_clean"
IDENTIFIER_COLUMN = "customer_tweet_id"


def load_corpus(dataset_path=DEFAULT_DATASET):
    """Load and validate a cleaned Spotify conversation corpus."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Retrieval dataset not found: {path}")
    try:
        frame = pd.read_csv(path, keep_default_na=False)
    except Exception as error:
        raise ValueError(f"Could not read retrieval dataset {path}: {error}") from error

    required = {CUSTOMER_COLUMN, REPLY_COLUMN}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Retrieval dataset is missing required columns: {sorted(missing)}")

    valid_customer = frame[CUSTOMER_COLUMN].astype(str).str.strip().ne("")
    valid_reply = frame[REPLY_COLUMN].astype(str).str.strip().ne("")
    frame = frame.loc[valid_customer & valid_reply].copy()
    if frame.empty:
        raise ValueError("Retrieval dataset contains no valid non-empty conversations.")
    return frame.reset_index(drop=True)


def retrieve_similar(message, top_k=5, dataset_path=DEFAULT_DATASET):
    """Return up to ``top_k`` similar conversations ordered by cosine score."""
    query = str(message or "").strip()
    if not query:
        raise ValueError("The query message cannot be empty.")
    try:
        top_k = int(top_k)
    except (TypeError, ValueError) as error:
        raise ValueError("top-k must be a positive integer.") from error
    if top_k <= 0:
        raise ValueError("top-k must be a positive integer.")

    corpus = load_corpus(dataset_path)
    if top_k > len(corpus):
        print(f"Warning: top-k={top_k} exceeds corpus size {len(corpus)}; returning all conversations.", file=sys.stderr)
        top_k = len(corpus)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(corpus[CUSTOMER_COLUMN].astype(str))
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()
    # Stable ordering makes repeated runs reproducible when scores tie.
    ranked_indices = sorted(range(len(scores)), key=lambda index: (-scores[index], index))[:top_k]

    results = []
    for index in ranked_indices:
        row = corpus.iloc[index]
        result = {
            "similarity_score": float(scores[index]),
            "customer_message": row[CUSTOMER_COLUMN],
            "spotify_reply": row[REPLY_COLUMN],
        }
        if IDENTIFIER_COLUMN in corpus.columns:
            result["source_conversation_id"] = row[IDENTIFIER_COLUMN]
        results.append(result)
    return results


def _print_results(query, results, dataset_path):
    print(f"Retrieval corpus: {dataset_path}")
    print(f"Query: {query}")
    print(f"Matches returned: {len(results)}")
    for number, result in enumerate(results, start=1):
        print("\n" + "=" * 72)
        print(f"MATCH {number} | similarity={result['similarity_score']:.4f}")
        if "source_conversation_id" in result:
            print(f"SOURCE ID: {result['source_conversation_id']}")
        print(f"CUSTOMER: {result['customer_message']}")
        print(f"SPOTIFY REPLY: {result['spotify_reply']}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Retrieve similar historical Spotify support conversations.")
    parser.add_argument("--message", required=True, help="New Spotify customer message to search for.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of matches to return (default: 5).")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Optional cleaned conversation CSV path.")
    args = parser.parse_args()
    try:
        results = retrieve_similar(args.message, args.top_k, args.dataset)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    _print_results(args.message.strip(), results, args.dataset)


if __name__ == "__main__":
    main()
