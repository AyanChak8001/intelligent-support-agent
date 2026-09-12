# Failure Analysis

This report is generated from existing measured artifacts. Interpretive statements are labeled separately from measured facts.

## 1. Rule baseline misclassifies 18 of 200 labeled examples

- Category: intent_classification
- Affected: 18 / 200 (9.0%)
- Evidence: C:\Users\ayanc\hiver-sde-assignment\results\rule_baseline_per_intent.csv, C:\Users\ayanc\hiver-sde-assignment\results\rule_baseline_confusion_matrix.csv, C:\Users\ayanc\hiver-sde-assignment\results\golden_evaluation_dataset.csv, C:\Users\ayanc\hiver-sde-assignment\results\end_to_end_agent_evaluation.csv

### Measured facts

- The confusion matrix contains 18 off-diagonal predictions out of 200 examples (9.0%).
- The largest individual confusion count is 1 for SUBSCRIPTION_PLAN predicted as CONTENT_AVAILABILITY_METADATA.

### Interpretation and limits

- The confusion artifact gives aggregate counts, not a causal feature analysis. End-to-end representative rows are a separate 20-example sample.

**Technical root cause:** The measured failure is in the deterministic rule classifier's intent decision boundary; the artifacts do not establish which rule or token caused each error.

**Impact:** Wrong intent selection can route retrieval and escalation logic toward mismatched support behavior.

**Recommended improvement:** Review the highest-volume confusion pairs with discriminative cue tests, then add regression cases before changing rules.

Representative examples:

- `spotify-1007217` — Help. Only some of my "Songs" are available offline on my Mac whereas all of them are available offline on my phone.
- `spotify-2774667` — Ill be sleeping early and hope to wake up with #reputation album at or at 😕

## 2. 12 of 20 end-to-end cases lack the basic grounding signal

- Category: retrieval
- Affected: 12 / 20 (60.0%)
- Evidence: C:\Users\ayanc\hiver-sde-assignment\results\end_to_end_agent_evaluation.csv, C:\Users\ayanc\hiver-sde-assignment\results\reply_quality_checks.csv

### Measured facts

- 12 of 20 end-to-end rows (60.0%) have basic_grounding_signal_present=false.
- The same sample has 10 blank grounding_score values; retrieval evidence count is present but does not itself prove useful grounding.

### Interpretation and limits

- Missing grounding_score is a measured limitation of the output signal, not proof that retrieval returned irrelevant documents in every row.

**Technical root cause:** The generation/evaluation pipeline does not produce a grounding score for fallback paths, so retrieval usefulness is not demonstrated for those cases.

**Impact:** The agent may expose generic responses without a measured connection to retrieved support evidence.

**Recommended improvement:** Add retrieval diagnostics such as evidence relevance and action-support coverage, and gate customer-facing handling on validated evidence quality.

Representative examples:

- `spotify-1601184` — Would really rather you just put in a bug report w/ devs. I love Spotify but this volume issue honestly kills it
- `spotify-1784734` — Please contact me about problems with app for Android. Thx
- `spotify-1906057` — Haha, there are thousands of them. Thanks for your attention, but I thought streaming and Clouds should work in other way 😐

## 3. Deterministic reply fallback is used in 10 of 20 end-to-end cases

- Category: reply_generation
- Affected: 10 / 20 (50.0%)
- Evidence: C:\Users\ayanc\hiver-sde-assignment\results\end_to_end_agent_evaluation.csv, C:\Users\ayanc\hiver-sde-assignment\results\reply_quality_checks.csv, C:\Users\ayanc\hiver-sde-assignment\results\llm_judge_results.csv

### Measured facts

- 10 of 20 rows (50.0%) have fallback_used=true.
- 8 fallback rows were AUTO_HANDLE and therefore returned a customer-facing reply; the other 2 were escalated with an internal draft.

### Interpretation and limits

- The deterministic checks identify fallback usage, but do not independently prove that every fallback reply was low quality. Human/LLM scores cover 16 examples, not all 20 end-to-end rows.

**Technical root cause:** No supported action was extracted for these rows, so generate_reply used its safe information-gathering fallback.

**Impact:** Fallback replies are less specific and may fail to advance troubleshooting despite being non-empty and privacy-safe.

**Recommended improvement:** Expand safe action extraction and add intent-specific fallback templates; retain escalation when the system lacks supportable next steps.

Representative examples:

- `spotify-1601184` — Would really rather you just put in a bug report w/ devs. I love Spotify but this volume issue honestly kills it
- `spotify-1784734` — Please contact me about problems with app for Android. Thx
- `spotify-1906057` — Haha, there are thousands of them. Thanks for your attention, but I thought streaming and Clouds should work in other way 😐

## 4. Escalation is concentrated in low-confidence or OTHER_UNCLEAR paths

- Category: escalation_behavior
- Affected: 4 / 20 (20.0%)
- Evidence: C:\Users\ayanc\hiver-sde-assignment\results\end_to_end_agent_evaluation.csv, C:\Users\ayanc\hiver-sde-assignment\results\escalation_cases.csv

### Measured facts

- 4 of 20 rows (20.0%) were escalated.
- 3 of those 4 escalations (75.0%) had Low confidence or predicted intent OTHER_UNCLEAR.
- One escalated row had a specific true intent but was predicted as OTHER_UNCLEAR.

### Interpretation and limits

- Escalation is a safety behavior, so the artifact does not establish that all escalations are failures. The specific-intent/OTHER_UNCLEAR row identifies a measurable routing opportunity, not proof that the policy itself is incorrect.

**Technical root cause:** The observed escalation policy escalates OTHER_UNCLEAR, low-confidence, and account/security cases; one sample row reached this path after an intent mismatch.

**Impact:** Customers receive no customer-facing draft on escalation, increasing handoff dependence and response latency.

**Recommended improvement:** Measure escalation precision and resolution outcomes by reason, and improve classification/evidence before relaxing escalation safeguards.

Representative examples:

- `spotify-1964335` — thinks it's funny to try and kick me out of my account and not allow me to play music. literally about to throw hands. not cool.
- `spotify-2582042` — You vanished???
- `spotify-2774667` — Ill be sleeping early and hope to wake up with #reputation album at or at 😕
- `spotify-396203` — Just sent you a DM

## 5. LLM and human overall scores disagree on 9 of 16 reviewed replies

- Category: human_llm_disagreement
- Affected: 9 / 16 (56.25%)
- Evidence: C:\Users\ayanc\hiver-sde-assignment\results\llm_judge_results.csv, C:\Users\ayanc\hiver-sde-assignment\data\evaluation\human_reply_evaluation_template.csv, C:\Users\ayanc\hiver-sde-assignment\results\human_llm_agreement.json

### Measured facts

- Overall exact agreement is 0.4375 across 16 matched examples.
- Overall MAE is 1.1875; the average signed difference (LLM minus human) is -1.1875.
- The LLM scored lower than the human on 9 of 16 overall scores.

### Interpretation and limits

- This is disagreement with one human evaluator, not evidence that either evaluator is correct. The 16-example human-reviewed sample is small and homogeneous; weighted kappa is undefined because human scores have no variance.

**Technical root cause:** The two evaluators apply different quality judgments to the same generated replies; the artifacts cannot identify whether rubric interpretation, calibration, or reply ambiguity is the dominant cause.

**Impact:** LLM-based quality monitoring may rank replies differently from the human benchmark, especially for overall quality.

**Recommended improvement:** Calibrate the judge on adjudicated examples, add explicit anchor examples to the rubric, and use disagreement review rather than treating the LLM score as ground truth.

Representative examples:

- `spotify-1973475` — how do I make my account a student one?
- `spotify-510537` — if I have a payment pending and my renewal is 12/10 and I just cancelled premium, will I get my money back from pending??
- `spotify-1601184` — Would really rather you just put in a bug report w/ devs. I love Spotify but this volume issue honestly kills it

## Sample-size limitations

- Intent baseline metrics cover 200 reviewed golden examples; end-to-end, escalation, and reply checks cover only 20 sampled examples.
- Human/LLM agreement covers 16 matched examples and one human evaluator.
- Aggregate confusion counts do not identify causal classifier rules, and deterministic quality checks do not replace human reply-quality review.
