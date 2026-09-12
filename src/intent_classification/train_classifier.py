"""Assess whether a legitimate separate labeled training set exists.

No classifier is trained unless there are eligible non-golden labels. This
repository currently has none, so the script writes a transparent assessment.
"""

import json
from pathlib import Path

import pandas as pd

from baseline_majority import DEFAULT_SOURCES, _golden_ids, _source_ids, TAXONOMY

ROOT = Path(__file__).resolve().parents[2]


def assess_training_data():
    golden_ids = _golden_ids()
    details = []
    eligible_total = 0
    eligible_labels = set()
    for path in DEFAULT_SOURCES:
        if not path.exists():
            continue
        frame = pd.read_csv(path, keep_default_na=False)
        column = "final_intent" if "final_intent" in frame else "intent" if "intent" in frame else None
        if column is None:
            details.append({"path": str(path), "reason": "No label column"})
            continue
        mask = frame[column].isin(TAXONOMY)
        labeled = frame.loc[mask].copy()
        ids = _source_ids(labeled)
        if "example_id" in labeled:
            eligible = labeled.loc[~labeled["example_id"].isin(golden_ids)]
        elif "customer_tweet_id" in labeled:
            ids_series = labeled["customer_tweet_id"].map(lambda value: f"spotify-{int(float(value))}")
            eligible = labeled.loc[~ids_series.isin(golden_ids)]
        else:
            eligible = labeled.iloc[0:0]
        eligible_total += len(eligible)
        eligible_labels.update(eligible[column].unique())
        details.append({"path": str(path), "label_column": column, "labeled_rows": len(labeled), "eligible_non_golden_rows": len(eligible), "eligible_intents": sorted(eligible[column].unique().tolist())})
    sufficient = eligible_total >= 30 and len(eligible_labels) >= 2
    return {"trained": False, "sufficient_separate_labeled_data": sufficient, "eligible_rows": eligible_total, "eligible_intents": sorted(eligible_labels), "details": details, "reason": "No legitimate separate labeled training data is available; no TF-IDF/logistic-regression model was trained." if not sufficient else "Training assessment passed; model training is intentionally not invoked by this assessment script."}


def main():
    result = assess_training_data()
    output = ROOT / "results" / "training_data_assessment.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(output)


if __name__ == "__main__":
    main()
