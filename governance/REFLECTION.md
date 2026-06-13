# Reflection — Diabetes 30-Day Readmission System

A short, honest retrospective: the trade-offs we made, what would change before this ran
on real patients, and the limits of the model itself.

## What the project actually optimized for

The model is ~20% of the work; the system around it is the rest, and that framing drove
every decision. The hard parts were never "which algorithm" — XGBoost vs LightGBM barely
moved the needle (val PR-AUC ~0.236 vs the logistic baseline, on a genuinely hard signal).
The parts that mattered were the ones that quietly break models in production: splitting by
patient so the metrics weren't a lie, calibrating probabilities so a "30%" means 30%,
sharing one feature transform between training and serving, and being able to *see* the
model once it's live. Those are the decisions documented in `docs/DESIGN_NOTES.md`, and
they're the transferable lessons.

## Key trade-offs

- **Recall over precision.** At a $300 follow-up vs a $15k readmission, we tuned the
  threshold to catch 70% of readmissions while accepting that ~5 in 6 flags are
  "wasted." That is correct *only* because the action is a cheap, non-punitive phone
  call. If the intervention were scarce or carried any downside for the patient, this
  trade-off would be wrong and the threshold would have to move sharply.
- **A modest model, honestly reported.** PR-AUC 0.24 on 11% prevalence is a real but
  limited signal — prior inpatient visits and discharge disposition do most of the work.
  We resisted dressing this up with accuracy (89% by predicting "never"). A weaker model
  reported honestly is more useful than a strong-looking one that misleads.
- **Calibration cost us nothing on ranking and bought us a valid cost threshold** — but
  it added a 3-model ensemble and a grouped-CV calibration step. Worth it: the entire
  cost-based threshold is meaningless on uncalibrated scores.
- **Live `/drift` endpoint over a batch job.** Simpler to demo and self-contained in the
  image, at the cost of computing a heavy Evidently report inside the serving process.
  In production this belongs in a separate worker, not the request path.
- **Self-contained local stack over cloud-first.** Docker Compose (API + Prometheus +
  Grafana) proves the whole system on one machine before any cloud spend; Cloud Run is
  the last step, not the first.

## What I'd change before real deployment

1. **Retrain on current data.** 1999–2008 / ICD-9 data is a teaching set, not a clinical
   one. Nothing else matters until this is done on a modern, ICD-10 population.
2. **Close the label loop.** Today we have no real outcome feedback; the retraining
   trigger's "recall < 0.55 on newly labelled data" rule assumes a pipeline that captures
   actual 30-day readmissions and feeds them back. That pipeline is the highest-value
   missing piece.
3. **Act on fairness, don't just report it.** v1 *discloses* the race (small-subgroup) and
   age (young-patient) gaps. Production would either collect enough data to make those
   subgroups reliable, adopt per-group thresholds, or scope young patients out — each
   validated against labelled outcomes, not chosen by eye.
4. **Harden the ops path.** Wire Alertmanager to a real channel (the rules already fire);
   move drift to a scheduled worker; add request auth and rate limiting; persist the audit
   log to durable storage for true lineage rather than a bind-mounted file.
5. **Add social-determinant features** beyond payer code — they're among the strongest
   real-world readmission predictors and are largely absent here.

## Limits of the model itself

- It predicts **risk, not need or benefit** — a high score doesn't mean an intervention
  will help that patient.
- It is **most reliable for the 50–90 bulk** and weak-to-unstable for young and rare
  subgroups.
- It depends heavily on **prior-utilization features**, so a patient with no recorded
  history (new to the system) is scored on thin evidence.
- Its precision is low by design; it is a **triage aid**, not a diagnosis.

## What I'd keep

The structural choices held up: patient-grouped splitting, one shared feature transform,
calibrated probabilities tied to an explicit cost model, model + spec exported as a matched
pair, every prediction logged with its version, and a model card / fairness audit generated
from code rather than written by hand. If I rebuilt this for a different domain tomorrow,
those are the patterns I'd carry over unchanged.
