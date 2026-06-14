# Model Evaluation Report

**Model:** xgboost + isotonic calibration · **Operating threshold:** 0.1 (flag a patient when risk ≥ 0.1).

## How the care team uses this
At discharge, each patient gets a **calibrated 30-day readmission risk** (an honest probability). The team proactively follows up with every patient **at or above the flag threshold (0.1)**. The threshold is not arbitrary — it falls out of the cost trade-off below, and each prediction comes with its top contributing factors so a clinician can sanity-check *why* before acting.

## The decision rule (cost assumptions)
- A 30-day readmission costs about **$15,000**.
- A follow-up intervention costs about **$300**.
- About **20.0%** of readmissions are preventable with timely follow-up.

Flag when the expected *preventable* cost beats the intervention cost:

&nbsp;&nbsp;`risk × 0.2 × $15,000 > $300`  →  **risk ≥ 0.1**.

## Performance on held-out test data
Computed **once**, on patients the model never saw during training or tuning — the honest estimate of real-world performance.

| metric | value | what it means |
|---|---|---|
| Recall @ 0.1 | **70.0%** | share of true 30-day readmissions the model flags |
| Precision @ 0.1 | 16.8% | share of flagged patients who are readmitted (low by design) |
| Flag rate | 47.2% | share of patients recommended for follow-up |
| PR-AUC | 0.244 | ranking quality at ~11% prevalence (vs 0.11 baseline) |
| ROC-AUC | 0.688 | overall ranking quality |
| Brier score | 0.095 | calibration error — lower is better (0.101 = base-rate bar) |
| Net savings | **$96,103 / 1,000 patients** | under the cost model above |

**Why precision is low on purpose:** a wasted follow-up call costs ~$300; a *missed* readmission costs ~$15,000. We deliberately favour catching readmissions (high recall) over avoiding false alarms.

## Threshold sensitivity
The flag threshold shifts if the cost assumptions change — useful when adapting the model to a different unit or budget:

| preventable fraction | readmission cost | optimal threshold |
|---|---|---|
| 10.0% | $10,000 | 0.3 |
| 10.0% | $15,000 | 0.2 |
| 10.0% | $20,000 | 0.15 |
| 20.0% | $10,000 | 0.15 |
| 20.0% | $15,000 | 0.1 |
| 20.0% | $20,000 | 0.075 |
| 30.0% | $10,000 | 0.1 |
| 30.0% | $15,000 | 0.067 |
| 30.0% | $20,000 | 0.05 |

The shipped threshold (**0.1**) corresponds to the highlighted assumptions (20.0% preventable, $15,000 cost).

*Generated from `src/models/final_model_report.json` by `src/governance/eval_report.py`.*