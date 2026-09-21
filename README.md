# Hiver AI Support Agent

## Project Overview

An explainable customer-support agent that processes customer messages, identifies support intent, retrieves relevant historical conversations, generates grounded responses, and determines whether a case can be handled automatically or should be escalated to a human.

The system is designed around deterministic, explainable components rather than opaque end-to-end generation. Intent classification uses rule-based logic, retrieval uses TF-IDF with cosine similarity, and response generation extracts supportable next-step patterns from relevant historical conversations rather than directly copying previous replies or relying on an external LLM.

The project also includes an evaluation framework with a human-reviewed dataset, automated quality checks, escalation analysis, and structured evaluation outputs for measuring system behavior and identifying failure cases.

## Problem Framing

The goal of this project is not simply to maximize intent-classification accuracy. For a Spotify-style customer-support agent, a good system should balance correctness, helpfulness, safety, and appropriate escalation.

I define a good support interaction as one where the system:

- Correctly identifies the customer's primary support need.
- Uses relevant historical support evidence rather than inventing unsupported instructions.
- Provides a clear and useful next step when the case can be handled safely.
- Avoids exposing personal information or requesting unnecessary account details.
- Escalates security-sensitive, unclear, or insufficiently grounded cases instead of confidently guessing.
- Produces responses with an appropriate and professional support tone.

The system is therefore evaluated as a pipeline rather than only as a classifier. Intent accuracy is useful, but retrieval quality, reply grounding, fallback behavior, escalation decisions, and human evaluation are also important parts of support quality.

### What I Intentionally Did Not Build

This assignment focuses on an explainable prototype and evaluation workflow rather than a production-ready customer-support platform. I intentionally did not build:

- Real Spotify account access or integration with Spotify production systems.
- Autonomous account changes, subscription changes, refunds, or security actions.
- Authentication, user management, or a customer-facing web interface.
- A fine-tuned supervised classifier trained on the golden evaluation labels, because that would create evaluation leakage.
- A production LLM reply-generation system; customer replies remain deterministic and grounded in retrieved historical evidence.
- Fully autonomous handling of account-access and security-sensitive cases.
- Production deployment, monitoring infrastructure, or real downstream ticket-resolution tracking.

These choices keep the prototype focused on the assignment's core problem while avoiding unsupported production claims.

## Architecture

```text
Customer Message
      ↓
Intent Classification
      ↓
Retrieval of Similar Historical Cases
      ↓
Reply Generation
      ↓
Escalation Decision
      ↓
AUTO_HANDLE or ESCALATE
```

Major runtime modules:

- `src/intent_classification/suggest_intents.py` and `baseline_rules.py`: fixed 11-intent taxonomy and explainable rule scoring.
- `src/retrieval/retrieve_similar.py`: cleaned-corpus loading, TF-IDF unigram/bigram vectors, cosine similarity, and stable top-k ranking.
- `src/reply_generation/generate_reply.py`: sanitizes historical replies, extracts safe supported actions, and creates deterministic grounded or fallback drafts.
- `src/escalation/decide_escalation.py`: applies security, unclear-intent, confidence, and retrieval safeguards.
- `src/support_agent.py`: integrates classification, retrieval, reply generation, and escalation. Escalated drafts remain internal.

Evaluation modules:

- `src/evaluation/evaluate_system.py`: full intent metrics and fixed-seed end-to-end evaluation.
- `src/evaluation/human_evaluation.py`: human-evaluation template generation, interactive review, and validation/aggregation.
- `src/evaluation/llm_judge.py`: optional Gemini structured-output judging and human agreement analysis.
- `src/evaluation/analyze_failures.py`: read-only analysis of measured failure patterns.

## Repository Structure

```text
data/raw/                 Original raw conversation data
data/processed/           Cleaned conversation and retrieval corpora
data/evaluation/          Labeling artifacts, guidelines, and human-review sheet
src/intent_classification Intent taxonomy, rules, suggestion and validation tools
src/retrieval/            TF-IDF retrieval
src/reply_generation/     Deterministic grounded reply generation
src/escalation/           Escalation policy
src/evaluation/           Automated, human, LLM, and failure evaluations
results/                  Reproducible evaluation outputs and reports
```

Important artifacts include `results/golden_evaluation_dataset.csv`, `results/golden_evaluation_metadata.json`, `data/evaluation/intent_suggestions_review_progress.csv`, `data/evaluation/human_reply_evaluation_template.csv`, and the reports under `results/`.

## Setup

From the repository root on Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The checked-in dependency configuration includes the official `google-genai` SDK. Gemini is optional for the core agent and deterministic evaluation. It is required only for the LLM-as-a-Judge command.

Configure the key through a secret manager or process environment without committing it or printing it. The following only verifies presence; set the variable through your local secret-management workflow:

```powershell
# Set GEMINI_API_KEY from your local secret manager before this check.
if ($env:GEMINI_API_KEY) { Write-Output "GEMINI_API_KEY is configured" }
```

The judge reads only `GEMINI_API_KEY` and never writes it to output artifacts.

## How to Run

Use `python` below, or `.\venv\Scripts\python.exe` when the virtual environment is not activated.

Validate the intent-suggestion artifacts:

```powershell
python src/intent_classification/validate_intent_suggestions.py
```

Retrieve similar historical cases:

```powershell
python src/retrieval/retrieve_similar.py --message "My downloaded songs are missing" --top-k 3
```

Generate a deterministic reply draft:

```powershell
python src/reply_generation/generate_reply.py --message "My downloaded songs are missing" --top-k 3
```

Make an escalation decision:

```powershell
python src/escalation/decide_escalation.py --message "Someone is using my account" --top-k 3
```

Run the integrated support agent:

```powershell
python src/support_agent.py --message "My downloaded songs are missing" --top-k 3
```

Run the deterministic automated evaluation. Defaults are the 20-example end-to-end sample, seed 42, and top-k 3:

```powershell
python src/evaluation/evaluate_system.py
```

Generate a human reply-evaluation template before scoring. Use a new output path so a completed human worksheet is not overwritten:

```powershell
python src/evaluation/human_evaluation.py generate-template --output data/evaluation/human_reply_evaluation_template.new.csv
```

Validate completed human reviews and regenerate their summary:

```powershell
python src/evaluation/human_evaluation.py validate
```

Run interactive human review only when additional human scoring is intended:

```powershell
python src/evaluation/human_evaluation.py interactive
```

Run the final fixed-model Gemini judge. The final experiment used `gemini-3.5-flash`; do not change the model when resuming quota-limited requests:

```powershell
python src/evaluation/llm_judge.py judge --model gemini-3.5-flash
python src/evaluation/llm_judge.py judge --resume --model gemini-3.5-flash
```

Generate human/LLM agreement from the completed judge results:

```powershell
python src/evaluation/llm_judge.py agreement
```

Generate the read-only failure analysis:

```powershell
python src/evaluation/analyze_failures.py
```

The evaluation and human-template commands write their documented result files. The analysis and agreement commands do not modify source datasets, labels, or human scores.

## Evaluation Results

All values below are copied from the completed artifacts under `results/`; they are measured results, not estimates.

### Intent Classification

The rule baseline was evaluated on all 200 reviewed golden examples:

- Accuracy: `0.91`
- Macro F1: `0.8985921479044575`
- Weighted F1: `0.9112696760014991`
- Dataset size: `200`

The repository also reports a trivial constant-class sanity check in `results/majority_baseline_evaluation.json`:

- Policy: predict the most frequent `final_intent` for every row.
- Selected class: `APP_TECHNICAL_COMPATIBILITY` (39 of 200 labels).
- Accuracy: `0.195`
- Macro F1: `0.0296690756941803`
- Weighted F1: `0.06364016736401674`
- Per-intent metrics: included in the JSON artifact; the majority class has recall `1.0`, precision `0.195`, and F1 `0.3263598326359833`; every other intent has precision, recall, and F1 `0.0`.

This is a descriptive sanity-check baseline, not an independent supervised experiment: the same golden evaluation labels select the constant class and assess it. The independent majority implementation found no separate eligible labeled rows after excluding golden IDs, so its availability remains `false` in `intent_classification_evaluation.json`.

### End-to-End Evaluation

The deterministic end-to-end evaluation used 20 examples sampled with seed `42` and retrieval top-k `3`:

- `AUTO_HANDLE`: 16
- `ESCALATE`: 4
- Fallback usage: 10/20
- Basic grounding signal present: 8/20
- Basic grounding signal absent: 12/20
- Non-empty auto-handled reply check: 20/20 passed
- No obvious URL/email leak check: 20/20 passed
- No obvious account-identifier leak check: 20/20 passed
- No customer reply on escalation check: 20/20 passed

The automated checks are safety and structural checks; they do not establish human helpfulness or correctness.

Retrieval overlap controls removed 213 rows with the 200 golden IDs from the temporary evaluation corpus. The corpus contained 221 exact customer-message overlaps, so broader overlap limitations remain.

### Human Evaluation

Human evaluation covered 16 generated customer-facing replies, scored by one human evaluator on six criteria using integer scores from 1 to 5.

Average scores:

- Helpfulness: `3.0`
- Correctness: `4.0`
- Relevance: `4.0`
- Tone: `5.0`
- Safety: `5.0`
- Overall: `4.0`

All 16 reviews were complete and valid. The human score distribution was homogeneous: every row received the same score for each criterion.

### LLM-as-a-Judge

The final experiment used one fixed model, `gemini-3.5-flash`, and completed 16/16 structured judgments. Average scores were:

- Helpfulness: `2.6875`
- Correctness: `3.6875`
- Relevance: `3.5`
- Tone: `4.4375`
- Safety: `5.0`
- Overall: `2.8125`

Gemini API quota limits required same-model retries. The final CSV contains exactly one model for all 16 rows.

### Human–LLM Agreement

Agreement was calculated over all 16 matched, valid examples. Signed difference is LLM minus human.

| Criterion | Exact agreement | MAE | Average signed difference |
|---|---:|---:|---:|
| Helpfulness | 0.1875 | 0.9375 | -0.3125 |
| Correctness | 0.1875 | 1.1875 | -0.3125 |
| Relevance | 0.0625 | 1.375 | -0.5 |
| Tone | 0.625 | 0.5625 | -0.5625 |
| Safety | 1.0 | 0.0 | 0.0 |
| Overall | 0.4375 | 1.1875 | -1.1875 |

Weighted Cohen’s Kappa is undefined and therefore reported as `null`: all human ratings for every criterion had no variance.

### Failure Analysis: Top 5 Failure Modes

The following failure modes were identified from the measured evaluation artifacts. The hypotheses below are engineering interpretations rather than directly proven causal findings.

| Failure mode | Measured result | Representative example/pattern | Hypothesis |

| Intent classification errors | 18/200 (9%) | Confusion between overlapping customer-support categories | Rule-based keyword and cue matching can be brittle when messages contain multiple intents or ambiguous wording. |
| Missing grounding signal | 12/20 (60%) | Retrieved evidence existed but did not provide a strong supported next step | Lexical TF-IDF similarity does not guarantee that retrieved conversations contain actionable guidance. |
| Reply-generation fallback | 10/20 (50%) | Relevant retrieval evidence lacked a recognized safe action pattern | The deterministic action extractor has limited coverage and intentionally avoids inventing unsupported instructions. |
| Escalation concentration | 4/20 (20%) | Most escalations occurred in low-confidence or `OTHER_UNCLEAR` paths | Ambiguous intent classification and conservative routing policies increase human escalation for uncertain cases. |
| Human/LLM disagreement | 9/16 (56.25%) on overall score | Human overall scores were higher than the fixed-model LLM judge on many examples | The human evaluator and LLM judge appear to apply different standards for reply helpfulness and overall quality. |

For the full evidence, representative examples, detailed interpretations, and recommended improvements, see `results/failure_analysis.md`.

# What is misleading about my headline number?

The headline intent accuracy is `0.91` on 200 reviewed golden examples. It measures whether the rule baseline predicted the reviewed 11-intent label for each example. It does not measure complete support-agent correctness: a correct intent can still retrieve weak evidence, produce a generic fallback, or make an unhelpful reply, and an incorrect intent can affect retrieval and escalation routing.

Several factors make this number easy to overinterpret:

- The rule baseline is not a blind benchmark. Its rules were developed after inspecting the task data and evaluation workflow, so the `0.91` may be optimistic. The result is also a single deterministic rule system, not evidence that a learned classifier would generalize.
- The full intent evaluation has 200 examples, but the end-to-end agent evaluation uses only 20 examples sampled with seed `42`. Its results were 16 `AUTO_HANDLE`, 4 `ESCALATE`, and 10 fallback cases; these are not full-dataset end-to-end measurements.
- Retrieval used a historical corpus with exact golden conversation IDs removed during evaluation, but the corpus still contained 221 exact customer-message overlaps. This limits claims about generalization to unseen customer language.
- The human reply evaluation covers only 16 replies and one evaluator. All human scores were homogeneous by criterion, which is why Cohen’s Kappa is undefined rather than a meaningful agreement estimate.
- The Gemini judge is not ground truth. It is a second evaluator with its own calibration and model behavior. Its 16-example agreement with the human scores should be read as a comparison study, not as an objective replacement for human judgment.

Therefore, `0.91` is a useful measured intent-classification result under this evaluation setup, but it is not an end-to-end support-quality score, a production accuracy guarantee, or proof that the agent resolves customer issues correctly.

## Limitations

- The end-to-end, escalation, and reply checks use only a 20-example sample; intent metrics use 200 examples.
- Human/LLM agreement uses 16 replies and one human evaluator.
- Human ratings are homogeneous, which makes Cohen’s Kappa undefined and limits calibration conclusions.
- The retrieval corpus has exact customer-message overlaps with the golden data even though exact golden IDs were removed during evaluation.
- TF-IDF is lexical and may miss semantic similarity, paraphrases, spelling variation, and context.
- Rule-based classification is explainable but brittle around overlapping or ambiguous cues.
- Reply generation is deterministic, concise, and retrieval-grounded, but can fall back to generic information gathering when no supported action is extracted.
- Security cases always escalate by policy; this improves caution but reduces automation.
- The LLM judge is an evaluator, not ground truth. Its scores and agreement can vary with model availability, quota behavior, model version, and rubric interpretation.
- The rule baseline was developed after inspecting the evaluation workflow, so its benchmark may be optimistic rather than blind.

## Future Improvements

Based on the observed evaluation results and failure analysis, potential improvements include:

Add regression tests for recurring intent-confusion cases before modifying classification rules.
Evaluate semantic retrieval alongside TF-IDF to determine whether evidence relevance improves on a leakage-controlled corpus.
Add retrieval diagnostics to measure evidence relevance and action-support coverage.
Expand safe action extraction and introduce intent-specific fallback strategies for common support scenarios.
