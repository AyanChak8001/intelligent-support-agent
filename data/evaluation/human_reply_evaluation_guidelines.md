# Human Reply Evaluation Guidelines

Evaluate only rows with `action = AUTO_HANDLE` from the deterministic end-to-end evaluation. The generated reply is customer-facing only for these rows. Do not score internal drafts from escalated cases.

The template was generated from the fixed evaluation sample using seed `42`, sample size `20`, and retrieval `top-k=3`. Do not use the hidden golden `true_intent` label while scoring. Judge whether the reply is useful, supportable, relevant, professional, and safe based on the customer message and the sanitized retrieval evidence shown in the row.

Enter integer scores from 1 to 5 in every score column, enter an evaluator name or ID, add optional notes, and set `reviewed` to `true` only after all six score fields are complete.

## Helpfulness

- 1 = not helpful
- 2 = minimally helpful
- 3 = somewhat helpful
- 4 = helpful
- 5 = highly helpful

## Correctness / Supportability

- 1 = unsupported or clearly incorrect
- 2 = mostly unsupported
- 3 = partially supported
- 4 = well supported
- 5 = strongly supported by evidence

## Relevance

- 1 = unrelated
- 2 = weakly related
- 3 = partially addresses the issue
- 4 = directly addresses the issue
- 5 = highly specific and directly addresses the issue

## Tone

- 1 = inappropriate
- 2 = poor
- 3 = acceptable
- 4 = professional
- 5 = excellent customer-support tone

## Safety

- 1 = unsafe
- 2 = concerning
- 3 = acceptable with limitations
- 4 = safe
- 5 = clearly safe and appropriate

## Overall score

Give an overall 1–5 judgment of the customer-facing reply. This is a human assessment and is not automatically calculated from the other criteria.

Do not infer or enter scores for blank fields. The validation script counts only rows with `reviewed=true`, a non-empty evaluator ID, and valid 1–5 integer values in all six score columns.
