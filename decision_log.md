# Engineering Decision Log

This log records the major design decisions reflected in the implementation and completed evaluations. Metrics are referenced from existing artifacts; no assignment requirement is inferred beyond the supplied project context. No original assignment document was found in the repository.

## 1. Fixed 11-intent taxonomy

**Context:** Customer-support messages span recurring areas such as billing, playback, account security, integrations, and unclear follow-ups.

**Decision:** Use the fixed 11-intent taxonomy defined in `src/intent_classification/suggest_intents.py`, including `OTHER_UNCLEAR`.

**Rationale:** The taxonomy gives the classifier, retrieval/generation policy, review workflow, and evaluation a shared vocabulary. `OTHER_UNCLEAR` provides a safe destination when a message lacks enough standalone context.

**Trade-offs or limitations:** A fixed taxonomy cannot represent every nuanced issue. Similar intents can overlap, and the measured rule baseline made 18 errors on 200 golden examples.

## 2. Explicit human review of the golden dataset

**Context:** Automatic intent suggestions are useful for triage but can be wrong, especially for ambiguous messages.

**Decision:** Keep explicit human review and use `final_intent` only after the review workflow records a reviewed label.

**Rationale:** The 200-example golden dataset is an evaluation reference, not automatically trusted model output. The labeling guidelines and progress artifacts make review status, overrides, and provenance visible.

**Trade-offs or limitations:** Human review costs time and introduces subjectivity. The resulting labels are still a reference standard rather than an absolute truth.

## 3. No supervised ML classifier was trained

**Context:** The available separately labeled data was insufficient after excluding golden evaluation IDs; the evaluation report records zero eligible separate training rows.

**Decision:** Use an explainable rule baseline and do not train a supervised classifier.

**Rationale:** Training on the golden evaluation examples would create leakage and invalidate the benchmark. The rule baseline works with the available evidence and exposes its decision logic.

**Trade-offs or limitations:** Rules are brittle and lexical. They do not learn from data and can miss paraphrases or resolve overlapping cues poorly.

## 4. TF-IDF retrieval

**Context:** The system needed retrieval over the cleaned historical Spotify support corpus without a separate semantic model or additional labeled training data.

**Decision:** Use TF-IDF with lowercase text, Unicode accent stripping, word unigrams/bigrams, sublinear term frequency, and cosine similarity.

**Rationale:** This is deterministic, inexpensive, inspectable, and reproducible. Stable tie ordering makes repeated retrieval results consistent.

**Trade-offs or limitations:** Lexical similarity is weaker for paraphrases and context. Evaluation found exact customer-message overlap in the broader corpus, so the evaluation removes exact golden IDs but does not eliminate every form of overlap.

## 5. Deterministic retrieval-grounded reply generation

**Context:** Historical replies can contain handles, links, account details, or case-specific language, and unrestricted copying could expose or invent information.

**Decision:** Sanitize retrieved replies, extract only safe supported next-step patterns, and compose a new deterministic draft. Use a generic information-gathering fallback when no supported action is available.

**Rationale:** This keeps the reply explainable and grounded in retrieved support evidence while avoiding direct historical-reply copying. The pipeline also keeps escalated drafts out of the customer-facing reply field.

**Trade-offs or limitations:** The approach is conservative and repetitive. Fallback generation occurred in 10 of 20 end-to-end cases, and the basic grounding signal was absent in 12 of 20 cases.

## 6. Security cases always escalate

**Context:** Account compromise, unauthorized use, and related security issues can require identity verification or account-specific intervention.

**Decision:** `ACCOUNT_ACCESS_SECURITY` is a risk-sensitive intent that always produces `ESCALATE`.

**Rationale:** A cautious handoff avoids unsafe account instructions and unsupported claims about account ownership or recovery.

**Trade-offs or limitations:** This reduces automation and may escalate cases that could eventually be resolved with safe self-service guidance. The end-to-end sample contained one such security escalation.

## 7. Dataset review flags are not direct production escalation triggers

**Context:** The dataset’s `requires_human_review` field identifies suggestions needing label review, not necessarily customers needing a support handoff.

**Decision:** Do not use `requires_human_review` as a direct production escalation trigger. The escalation adapter records the flag but explicitly reports that it was not used for escalation.

**Rationale:** Mixing annotation workflow state with runtime customer risk would conflate two different decisions and could over-escalate cases.

**Trade-offs or limitations:** Runtime escalation must rely on its own intent, confidence, evidence, and risk policy. Annotation uncertainty may still indicate areas worth improving offline.

## 8. Exact self-retrieval prevention during evaluation

**Context:** The golden examples are also related to the historical data source, so exact retrieval could make evaluation unrealistically easy.

**Decision:** During evaluation, remove rows with exact golden conversation IDs from a temporary retrieval corpus.

**Rationale:** This prevents direct self-retrieval while preserving the source corpus and all protected artifacts. The evaluation recorded 213 removed rows across 200 golden IDs.

**Trade-offs or limitations:** Exact customer-message overlap and broader near-duplicate overlap remain. The control improves but does not guarantee complete independence.

## 9. Gemini for LLM-as-a-Judge

**Context:** The assignment context required automated metrics plus an LLM judge and comparison with human evaluation.

**Decision:** Use the official `google-genai` SDK, `GEMINI_API_KEY`, structured JSON output, and six 1–5 reply-quality criteria.

**Rationale:** Gemini provides an independently implemented evaluator with schema validation. The judge receives only the customer message and generated reply, not human scores or hidden intent labels.

**Trade-offs or limitations:** API quotas, model availability, prompt interpretation, and evaluator bias affect results. The LLM is not ground truth and must be compared with human review.

## 10. One fixed Gemini model for the final experiment

**Context:** Initial quota recovery produced a historical mixed-model run. Different models were used across retries, making its agreement values unsuitable as one controlled experiment.

**Decision:** Regenerate the final results with one fixed `gemini-3.5-flash` model across all 16 examples. Resume retries are rejected if existing successful rows use another model.

**Rationale:** A single model makes the human-versus-LLM comparison internally consistent. The final run completed 16/16 judgments with exactly that model.

**Trade-offs or limitations:** The fixed model was subject to a 5-requests-per-minute quota, so same-model retries were required. A fixed model improves comparability but does not remove model bias or temporal API variability.

## 11. Separate human and LLM evaluation reporting

**Context:** Human review and LLM judging represent different evaluators with different calibration and possible biases.

**Decision:** Report human averages, LLM averages, and agreement metrics separately; do not combine them into a single quality score.

**Rationale:** Separation preserves provenance and makes disagreement visible. In the completed experiment, overall exact agreement was 0.4375 and MAE was 1.1875 across 16 matched examples.

**Trade-offs or limitations:** Agreement with one human evaluator does not establish correctness. The human ratings had no variance, so weighted Cohen’s Kappa was undefined.

## 12. Descriptive majority sanity-check baseline

**Context:** The assignment context calls for comparison with a trivial baseline, but the repository has no separate eligible labeled rows after excluding the golden evaluation IDs. The existing independent majority path therefore reports unavailable.

**Decision:** Add a constant-class majority sanity check selected from the 200 golden `final_intent` labels, while marking it as label-leaking and non-independent in every report.

**Rationale:** This provides a transparent class-distribution floor without falsely claiming a held-out supervised experiment. The deterministic policy selects `APP_TECHNICAL_COMPATIBILITY`, the most frequent class with 39/200 labels, and predicts it for every example.

**Trade-offs or limitations:** The same evaluation labels select and assess the class, so the result is descriptive rather than an unbiased benchmark. Its measured accuracy is `0.195`, macro F1 is `0.0296690756941803`, and weighted F1 is `0.06364016736401674`. A future independent labeled split is required for a legitimate held-out majority comparison.
