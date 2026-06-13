# Model Card — 30-Day Diabetic Readmission Risk

A care-team decision-support model that estimates, at discharge, each diabetic patient's
risk of being readmitted within 30 days, so follow-up effort can be targeted.

| | |
|---|---|
| **Model** | XGBoost (native categorical) + isotonic probability calibration |
| **Version** | `readmission-risk` v1 (MLflow registry) |
| **Type** | Binary classifier → calibrated probability + flag |
| **Operating threshold** | 0.10 (cost-based; see *Threshold*) |
| **Owners** | Project team (academic build) |
| **Last evaluated** | On the held-out test split, once, at finalization |

## Intended use

- **Intended:** flag discharged diabetic inpatients for proactive 30-day follow-up
  (calls, medication reconciliation, early appointments). The score and its top
  contributing factors are **decision support for clinicians**, not an automated action.
- **Users:** care-coordination / population-health teams.
- **Out of scope:** denying care or coverage, allocating cost/penalties to patients,
  any use on non-diabetic or non-inpatient populations, and any *autonomous* decision
  without a human in the loop. The model estimates *risk*, not clinical need or benefit.

## Training data

- UCI **Diabetes 130-US Hospitals (1999–2008)**: 101,766 encounters; after removing
  encounters that cannot be readmitted (expired / hospice discharge) and cleaning,
  **99,340 encounters / ~70k patients**.
- **Target:** `readmitted == "<30"` → 1 else 0; prevalence **~11.3%**.
- **Split:** by `patient_nbr` (GroupShuffleSplit 70/15/15) so no patient spans splits;
  CV tuning used `StratifiedGroupKFold`. Test split touched exactly once.
- Missingness treated as signal (`?`→`Missing`; a missing A1C/glucose test kept as its
  own "NotMeasured" category, never imputed to "normal"). `weight` dropped (~97% missing).
- **Known limitations of the data:** 1999–2008 US inpatient data, ICD-9 coding era — the
  population, coding, and care patterns differ from today; retraining on current data is
  required before any real deployment.

## Performance (held-out test split, n = 14,859)

| Metric | Value | Note |
|---|---|---|
| PR-AUC | **0.244** | vs 0.113 prevalence baseline — the headline metric at 11% positives |
| ROC-AUC | 0.688 | |
| Recall @ threshold 0.10 | **0.700** | 70% of true 30-day readmissions are flagged |
| Precision @ threshold 0.10 | 0.168 | most flags are not readmitted (accepted — see below) |
| Flag rate | 0.472 | share of patients recommended for follow-up |
| Brier score | 0.095 | beats the 0.101 base-rate Brier → calibrated |
| Net savings | ~$96k / 1,000 patients | under the cost model below |

**Accuracy is deliberately not reported as the headline** — at 11% prevalence a
"never readmit" model scores 89% accuracy while catching nobody.

### Threshold

Chosen from the precision–recall trade-off against an explicit cost model: a follow-up
intervention costs ~$300, a readmission ~$15,000, of which ~20% is preventable. Flagging
when `risk × 0.20 × $15,000 > $300` (i.e. risk ≳ 0.10) maximizes net savings. We
intentionally accept **low precision for high recall**: a wasted follow-up call is cheap;
a missed readmission is not.

## Explainability

- **Per-patient:** `/predict` returns the top contributing factors via XGBoost TreeSHAP
  (`pred_contribs`), signed and readable by a care team.
- **Global** (`governance/shap_global_importance.{md,png}`): the dominant drivers are
  **prior inpatient visits**, **discharge disposition**, **total prior visits**,
  **payer code**, and **primary diagnosis group** — consistent with the per-patient
  factors and the EDA (prior utilization is the strongest signal).

## Fairness

Fairlearn audit on the test split (`governance/fairness_report.md`), at threshold 0.10.
We weight **recall parity** most: a missed readmission is the harm the system exists to
prevent, so a recall gap means a group is protected less.

| Attribute | Recall gap (max−min) | Reading |
|---|---|---|
| **Gender** | 0.045 | Essentially equitable (F 0.72 / M 0.68). |
| **Race** | 0.218 | Driven by **small subgroups** (Asian n=94 recall 0.57, Hispanic n=307 recall 0.79). The two large groups are close (Caucasian 0.70, AfricanAmerican 0.72). |
| **Age** | 0.829 | Largely an artifact of the tiny `[0-10)` bracket (n=20, ~no positives → recall 0). There is also a **real trend**: older patients are flagged far more (selection rate 0.05 → 0.63) and recalled better — the model is most reliable for the 50–90 bulk and **unreliable for young patients**, who are rare in the data. |

**Findings, stated plainly:** the model is fair on gender; race disparities concentrate in
under-represented subgroups where estimates are noisy; age shows both a small-sample
artifact and a genuine reliability gap for young patients. None of these are mitigated in
v1 — they are **disclosed limitations**. Mitigation (e.g. per-group thresholds, reweighting,
or excluding age brackets too rare to support a reliable estimate) is future work and must
be validated against newly labelled data.

## Limitations

- **Low precision by design** — ~5 in 6 flags will not be readmitted; only deploy where
  the follow-up action is low-cost and non-punitive.
- **Stale data vintage** (1999–2008, ICD-9) — not fit for real clinical use without
  retraining on current data.
- **Small / rare subgroups** (young patients, Asian/Other/Missing race) have unstable and
  weaker performance.
- **No social-determinant features** beyond payer code; outcomes are hospital-coded only.
- Predicts **risk, not need** — a high score is not itself a clinical recommendation.

## Lineage & auditability

Every prediction is traceable:

- **Model version** — `readmission-risk` v1 in the MLflow registry; the version is
  returned by `/predict` and `/health` and written into every audit-log entry.
- **Data version** — raw + cleaned data and the cleaning/feature pipeline are
  DVC-tracked (`dvc.lock` pins data + code + output hashes together per git commit).
- **Code version** — the git commit of the deployed image identifies the exact code.
- **Per-request audit log** — `logs/predictions.jsonl` records timestamp, model version,
  inputs, risk, and flag for each scored request (Stage 5).

To reproduce a past prediction: check out the recorded git commit, `dvc pull` the pinned
data, load the recorded model version, and replay the logged inputs.

## Human-in-the-loop

The model is **advisory**. Concretely:

1. **No autonomous action** — a flag triggers a clinician review, never an automatic
   change to a patient's care or coverage.
2. **Borderline band** — predictions within roughly ±0.03 of the 0.10 threshold are
   surfaced as "uncertain" for explicit human judgement rather than treated as a hard
   yes/no.
3. **Explanations travel with the score** — the top SHAP factors are shown so the
   reviewer can sanity-check *why* a patient was flagged and override when the reasoning
   doesn't fit the clinical picture.
4. **Monitoring closes the loop** — drift (`/drift`) and the flag-rate alert tell
   operators when the model's behaviour has shifted enough to warrant re-review or
   retraining (see `monitoring/README.md`).

## Monitoring & retraining

Service + model metrics (Prometheus/Grafana), live data/prediction drift (Evidently
`/drift`), and alerts are described in `monitoring/README.md`, which also defines the
concrete retraining trigger (≥30% feature drift or prediction drift; recall < 0.55 on
newly labelled data; quarterly floor).
