# Diabetes 30-Day Readmission Prediction — End-to-End ML System

Predict each diabetic patient's risk of readmission within 30 days of discharge, so care
teams can target follow-up. Source brief: `Problem1 Student Handout.pdf` (read it before
changing scope). **The model is ~20% of the grade; the production system around it
(pipeline, deployment, monitoring, governance) is the other 80%.**

## Key decisions (agreed with the user — do not silently change)

| Decision | Choice | Why |
|---|---|---|
| Target | **Binary: `readmitted == "<30"` → 1, else 0** | Matches the business action (flag at discharge for 30-day follow-up); ~11% positive |
| Baseline | Logistic regression + class weighting | The bar to beat |
| Advanced model | **Train both XGBoost and LightGBM**, tune with CV, compare in MLflow, ship the winner | Strongest "baseline vs advanced" story |
| Split | **By `patient_nbr` (patient ID)**, train/val/test; test touched once at the end | Same patient appears in multiple rows — random row split = leakage |
| Headline metrics | PR-AUC, recall at fixed precision, calibration | Accuracy is a trap at 11% prevalence |
| Cloud | **GCP Cloud Run** (Artifact Registry + Cloud Build; Cloud Monitoring/Logging) | Simplest free tier; scales to zero |
| Serving | FastAPI `/predict` → risk score + top contributing factors, Dockerized, docker-compose with Prometheus + Grafana | Per handout |
| Drift | Evidently (data + prediction drift vs training distribution) | Per handout |
| Governance | Fairlearn fairness audit, SHAP explanations, audit logging, model card, MLflow registry lineage | Per handout |

## Data

- `data_folder/diabetic_data.csv` — 101,766 encounters (one row = one hospital stay), ~50 columns, target `readmitted` ∈ {`<30`, `>30`, `NO`}
- `data_folder/IDS_mapping.csv` — descriptions for `admission_type_id`, `discharge_disposition_id`, `admission_source_id`
- UCI "Diabetes 130-US Hospitals (1999–2008)" dataset

### Known traps (explicitly graded)
1. **Missing values are coded as `?`**, not blanks — treat as NaN on load.
2. **`weight` is ~97% missing**; `payer_code` and `medical_specialty` heavily missing. Decide drop vs impute and *write the justification down*.
3. **A missing A1C/glucose test is information** — keep "not measured" as its own category; never impute a "normal" value.
4. **Split by patient ID**, never randomly by row.
5. **State the target choice and why** (done — see decisions table).
6. Consider excluding encounters where readmission is impossible (e.g. `discharge_disposition_id` = expired/hospice) and document it.

## Environment

- Windows 11, PowerShell. Package/env manager: **uv** (0.11+), **Python 3.12** pinned.
- Project is uv-managed: deps live in `pyproject.toml` + `uv.lock` (committed); run everything
  with `uv run python ...` / `uv run pytest`; add deps with `uv add <pkg>`. Do not use pip
  directly or hand-edit a requirements.txt.
- **Docker Desktop is NOT installed yet** — required from Stage 4 onward; user must install it.
- git is available but the repo is **not initialized yet** — Stage 0 does `git init`. Never commit data files, secrets, keys, or credentials; `.gitignore` from day one.

## Target repo structure

```
├── CLAUDE.md / README.md
├── data_folder/            # raw data (gitignored), DVC-tracked
├── src/
│   ├── data/               # load, clean, validate (reproducible pipeline, not loose cells)
│   ├── features/           # feature engineering + feature log
│   ├── models/             # train, evaluate, threshold selection
│   └── api/                # FastAPI service
├── notebooks/              # EDA only — anything load-bearing graduates to src/
├── tests/
├── monitoring/             # prometheus.yml, grafana dashboards, evidently reports
├── governance/             # fairness audit, model card, audit-log sample, reflection
├── Dockerfile, docker-compose.yml
└── pyproject.toml, uv.lock     # uv-managed deps, Python 3.12 pinned
```

## Implementation plan — stage by stage

Work in order. Each stage ends with a runnable artifact and a short written record.

### Stage 0 — Project setup (prerequisite, not in handout)
1. `git init`, `.gitignore` (data, `.venv`, `mlruns/`, secrets, `__pycache__`).
2. `uv init` with Python 3.12 pinned (`.python-version`); `uv add` the ML stack: pandas, scikit-learn, xgboost, lightgbm, mlflow, fastapi, uvicorn, shap, fairlearn, evidently, dvc, pandera (or great-expectations), pytest, matplotlib. Commit `pyproject.toml` + `uv.lock`.
3. Create the folder skeleton above.

### Stage 1 — Data Engineering & Exploration
1. **Profile the raw data**: dtypes, per-column missingness (after `?`→NaN), cardinality, class balance (expect ~11% `<30`). Save the profile as an artifact.
2. **Reproducible cleaning pipeline** (`src/data/clean.py`, a function/script):
   - `?` → NaN on load.
   - Drop `weight` (~97% missing), `examide`/`citoglipton` (single-valued); decide and justify `payer_code`, `medical_specialty` (likely: keep as "Missing" category — missingness is informative).
   - Keep A1Cresult / max_glu_serum "None" as an explicit "not measured" category.
   - Map `admission_type_id` / `discharge_disposition_id` / `admission_source_id` via IDS_mapping; collapse NULL/Not Available/Not Mapped.
   - Filter out encounters that cannot be readmitted (expired/hospice discharge) — document.
3. **Define the target**: binary `<30` vs rest, implemented in one place, documented.
4. **Validation + versioning**: Great Expectations (or pandera) checks on the cleaned output; DVC-track raw + cleaned data (or a clearly documented re-runnable script).

### Stage 2 — Feature Engineering
1. **Group ICD-9 `diag_1/2/3` into clinical buckets** (circulatory, respiratory, digestive, diabetes, injury, musculoskeletal, genitourinary, neoplasms, other) — standard Strack et al. grouping.
2. **Encode + engineer**: total prior visits (`number_outpatient + number_emergency + number_inpatient`), count of medication changes, count of active diabetes meds, service-utilization features; ordinal-encode `age` brackets; one-hot or native-categorical for the rest (LightGBM can take categoricals natively — keep both paths possible).
3. **Feature log**: `src/features/FEATURES.md` — every feature, its source columns, and why it exists. Update it in the same commit as the code.

### Stage 3 — Modeling & Evaluation
1. **Split by `patient_nbr`** into train/val/test (e.g. 70/15/15, GroupShuffleSplit). Test set is touched exactly once, at the very end.
2. **Baseline**: logistic regression + `class_weight="balanced"` in a sklearn Pipeline. Log to MLflow.
3. **Advanced**: XGBoost and LightGBM; tune with grouped CV (`StratifiedGroupKFold`); handle imbalance via `scale_pos_weight` / class weights (justify vs resampling). Pick the winner on validation PR-AUC.
4. **Evaluate**: PR-AUC, ROC-AUC, recall at fixed precision (pick a precision the care team could live with), calibration curve + Brier score. **Accuracy is not the headline metric.**
5. **Operating threshold**: choose from the precision-recall trade-off tied to real costs (cost of follow-up call vs cost of missed readmission); write the justification.
6. **MLflow for everything**: every run logged (params, metrics, artifacts); winner registered in the MLflow model registry.

### Stage 4 — Package & Deploy (needs Docker Desktop installed)
1. Save winning pipeline (joblib); **FastAPI** app with `/predict` (JSON risk score + top contributing factors via SHAP values), `/health`, `/metrics`.
2. **Dockerfile** → build, run, hit the endpoint locally.
3. **docker-compose**: API + Prometheus + Grafana — whole stack on one machine first.
4. **GCP**: push image to Artifact Registry (Cloud Build), deploy to **Cloud Run**. Free tier + $300 credits; set a billing alert. No secrets in the repo.
5. **Rollback plan**: keep previous image tag; document the one-command redeploy of it.

### Stage 5 — Observability & Monitoring
1. **Service metrics**: latency, request count, error rate via Prometheus client in the API + Grafana dashboard (Cloud Monitoring once on Cloud Run).
2. **Model metrics**: log every prediction; track score distribution over time.
3. **Drift**: Evidently report comparing live/recent inputs to the training distribution (data drift + prediction drift); surface as report or dashboard panel.
4. **Alerting**: at least one alert that would notify a human (e.g. error rate or drift threshold).
5. **Retraining trigger**, concretely: e.g. "if Evidently dataset drift score > X, or recall on newly labelled data < Y, retrain."

### Stage 6 — Governance & Re-evaluation
1. **Fairness**: Fairlearn audit across age, gender, race — report per-group disparities (selection rate, recall), not just averages.
2. **Explainability**: SHAP global importance + per-patient explanations readable by a care team.
3. **Auditability**: log every scored request (inputs, score, model version, timestamp).
4. **Model card**: intended use, training data, performance, limitations, fairness findings.
5. **Lineage**: data version + code commit + model version per prediction (MLflow registry).
6. **Human-in-the-loop**: state how low-confidence/borderline cases get human review.
7. **Reflection (1–2 pages)**: trade-offs, what you'd change in production, model limits.

## Definition of done

Someone clones the repo, follows the README, rebuilds the data, retrains the model,
launches the service in one or two commands, hits the endpoint and gets a sensible risk
score with explanations, opens the dashboard, and reads the model card — **without asking
you anything**.

## Working agreements for this project

- Interactive style: discuss and agree on decisions before implementing them; when
  unsure, ask the user rather than assuming.
- Scripts over notebooks for anything load-bearing; notebooks are for EDA only.
- Every stage produces a written justification of its decisions (graded).
- Run Python via `uv run python ...`; manage deps via `uv add`. Never commit data or secrets.
