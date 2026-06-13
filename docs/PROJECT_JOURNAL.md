# Project Journal — What, Why, How

A detailed running record of each stage: what we did, why we did it, and how. Written
for learning — read alongside `docs/DESIGN_NOTES.md` (generic concepts) and
`CLAUDE.md` (the plan). Newest stage at the bottom.

---

## Stage 0 — Project Setup

### What we did
- Initialized a git repo with a `.gitignore` that excludes data, environments, model
  artifacts, and secrets from day one.
- Created a uv-managed Python 3.12 environment: `uv init`, `uv python pin 3.12`,
  `uv add <full ML stack>` — producing `pyproject.toml` (what we depend on) and
  `uv.lock` (exact pinned versions of all 234 resolved packages).
- Created the folder skeleton: `src/{data,features,models,api}`, `notebooks`, `tests`,
  `monitoring`, `governance`, `docs`.
- Set up the GitHub remote with a feature-branch workflow: nothing goes to `main`
  except by PR.

### Why
- **Lockfile over requirements.txt**: `pyproject.toml` says "pandas >= 2.3", the
  lockfile says "pandas 2.3.3 exactly, with these exact transitive deps". Anyone
  running `uv sync` gets a byte-identical environment — that's the foundation of the
  "someone clones your repo and reproduces everything" grading bar.
- **Python 3.12 not 3.14**: the ML stack (xgboost, shap, numba) publishes wheels for
  established versions first; 3.12 avoids fighting builds.
- **`.gitignore` before first commit**: it's much easier to never commit data/secrets
  than to scrub them from git history later.

### How (key commands)
```powershell
git init
uv init --python 3.12 --bare ; uv python pin 3.12
uv add pandas scikit-learn xgboost lightgbm mlflow fastapi uvicorn shap fairlearn evidently dvc pandera pytest ...
```

---

## Stage 1 — Data Engineering & Exploration

### What we did
1. **EDA notebook** (`notebooks/01_eda.ipynb`) answering eight specific questions,
   each tied to a decision we'd have to defend. Findings recorded in the final cell.
2. **Cleaning pipeline** (`src/data/clean.py`) — a script, not notebook cells —
   implementing six agreed rules, logging row counts at every step.
3. **Data contract** — a strict pandera schema enumerating all 48 output columns with
   dtypes, plus targeted value checks guarding each cleaning rule.
4. **Tests** (`tests/test_clean.py`) — five tests, one per rule family.
5. **Data versioning** — DVC tracks the raw CSVs (pointers in git, data outside);
   the cleaning step is a `dvc.yaml` pipeline stage so `dvc repro` rebuilds it when
   code or data changes. See DESIGN_NOTES entry 1 for the DVC mental model.

### Why — the decisions and their evidence
| Decision | Evidence from EDA |
|---|---|
| Binary target `<30` vs rest | the care team acts only on the 30-day risk; 11.16% positive — accuracy would be a useless metric |
| Split by patient, not row | 23.5% of patients have >1 encounter = **46.2% of all rows**; row-splitting would put the same patient in train and test |
| Drop `weight` | 96.9% missing — nothing to learn from |
| Keep "Missing" as a category for `medical_specialty`/`payer_code`/`race` | missingness is informative (missing-race group readmits at 8.3% vs 11.4% overall) |
| `A1Cresult`/`max_glu_serum` "None" → `NotMeasured`, never imputed | patients with **no** A1C test readmit *more* (11.4%) than any measured group (9.7–10.1%) — "not tested" is signal |
| Exclude expired/hospice discharges (ids 11,13,14,19,20,21) | readmission is impossible; 2,423 rows whose 1.8% "readmit rate" is noise |
| Drop 3 `gender="Unknown/Invalid"` rows | unusable for the fairness audit, statistically irrelevant |

### How — the non-obvious mechanics
- **The `"None"` trap**: pandas' default NA list includes the string `"None"`, so a
  plain `read_csv` silently destroys the not-measured signal. Fix:
  `read_csv(..., na_values=["?", ""], keep_default_na=False)` — only `?` and empty
  mean missing; `"None"` survives as a string we rename to `NotMeasured`.
- **Schema as tripwire**: pandera validates the cleaned output on every run with
  `strict=True` — a renamed/missing/extra column, a stray `?`, or an Expired row
  fails the pipeline loudly instead of corrupting downstream stages silently.
- **`IDS_mapping.csv` is three stacked tables** separated by blank lines — parsed
  into three dicts, then ids mapped to readable descriptions with
  NULL/Not Available/Not Mapped collapsed into one `Unknown` category.

Result: 101,766 → **99,340 rows (97.62% kept), 48 columns, 11.39% positive**.

---

## Stage 2 — Feature Engineering

### What we did
1. **`src/features/build_features.py`** — consumes `cleaned.parquet`, emits
   `features.parquet` (99,340 × 51), wired into `dvc.yaml` as the `features` stage.
2. **ICD-9 → clinical buckets**: `diag_1/2/3` (716–789 unique codes each) mapped to
   10 buckets (Circulatory, Respiratory, …, Missing); raw codes dropped.
3. **Engineered features**: `total_prior_visits`, `n_med_changes`, `n_active_meds`,
   `age_ordinal`, `diabetes_diag_any`.
4. **Rare-category lumping**: `medical_specialty` 70+ → 11 categories,
   `payer_code` 19 → 11 (everything under 1% of rows → `Other`).
5. **Feature log** (`src/features/FEATURES.md`): every feature, its definition, and
   why it exists — including what we deliberately did NOT include.
6. **Tests**: 15 parametrized unit tests for the ICD-9 bucketing function + 5
   sanity checks on the real output (schema, no leakage, plausible distributions).

### Why
- **Bucketing beats raw codes**: 700+ categories would explode one-hot width for the
  baseline and invite overfitting; ten clinically meaningful groups follow
  **Strack et al. 2014** (the original study on this dataset), making the choice
  citable rather than arbitrary. Verified: Circulatory is the biggest primary bucket
  (29.9%) — matches the top raw codes 428 (heart failure) / 414 (ischemic heart).
- **`age_ordinal` instead of one-hot age**: the brackets are *ordered* ([50-60) is
  between [40-50) and [60-70)); one-hot throws that ordering away, an ordinal 0–9
  keeps it.
- **Lumping at 1% is leakage-safe**: it uses only category *frequencies*, never the
  target, so computing it before the train/test split is fine. (Target-based feature
  decisions, by contrast, must happen inside the training fold only.)
- **Keep the 21 near-constant med columns**: low *univariate* correlation is not
  evidence of uselessness for tree models, which exploit interactions. SHAP will rank
  them after modeling; pruning before evidence would be unjustifiable in the write-up.
- **Drop `readmitted` here**: the 3-class column the target is derived from must never
  be visible to a model — that's leakage by definition. A schema check
  (`raw_label_removed`) enforces it permanently.
- **No encoding in this stage**: categoricals leave as pandas `category` dtype; each
  model pipeline encodes for itself (one-hot for logistic regression, native
  categorical splits for LightGBM/XGBoost). One feature table serves both models, and
  later the serving API can send raw-ish values through the same pipeline.

### How — the non-obvious mechanics
- **ICD-9 parsing**: codes are strings — mostly numeric (`"428"`, `"250.83"`) but
  also `V`/`E` prefixed supplementary codes. The bucketing function handles V/E first,
  then compares numerically. Mapping is computed once per *unique* code and applied
  via `.map()` — ~750 function calls instead of ~300k.
- **`n_med_changes`** = row-wise count of `Up`/`Down` across the 21 med columns
  (`meds.isin(["Up","Down"]).sum(axis=1)`); **`n_active_meds`** = count of `!= "No"`.
  `MED_COLS` is imported from `clean.py` — single source of truth for the list.
- **Module execution**: scripts that import from `src.*` must run as modules
  (`python -m src.features.build_features`), not as file paths — running by path puts
  the *script's folder* on `sys.path` instead of the project root, breaking imports.
  Both `dvc.yaml` stages use the `-m` form.
- **Consistency checks in the schema**: e.g. `total_prior_visits` must equal the sum
  of its three components — guards against a future edit breaking the definition.

Result: **99,340 rows × 51 columns**; identifiers + target + 13 numeric features +
35 categorical features, all `category`-typed, encoding deferred to model pipelines.

---

## Stage 3 — Modeling & Evaluation

### What we did
1. **Patient-grouped 70/15/15 split** (`src/models/split.py`, DVC `split` stage):
   GroupShuffleSplit on `patient_nbr`; 69,613 / 14,868 / 14,859 rows with target
   rates 11.4 / 11.6 / 11.3%. Four tests enforce zero patient overlap.
2. **Three models in MLflow** (`src/models/train.py`): logistic-regression baseline
   (one-hot + scaling + class weights) and XGBoost / LightGBM tuned with 20-candidate
   random search over 3 patient-grouped, stratified CV folds, optimizing PR-AUC.
3. **Winner finalized** (`src/models/finalize.py`): best XGBoost wrapped in isotonic
   calibration, cost-based threshold derived, model registered as
   **`readmission-risk` v1** in the MLflow registry, and the test set evaluated
   **once**. Full numbers in `src/models/final_model_report.json`.

### The results
| model (validation) | PR-AUC | ROC-AUC | recall@p≥30% | Brier |
|---|---|---|---|---|
| logreg baseline | 0.2212 | 0.6671 | 0.189 | 0.2248 |
| **xgb (winner)** | **0.2391** | **0.6803** | **0.216** | 0.2059 |
| lgbm | 0.2387 | 0.6753 | 0.215 | 0.2009 |
| xgb + isotonic calibration | 0.2356 | 0.6799 | 0.205 | **0.0969** |

Final **one-time test** (xgb calibrated): PR-AUC **0.2438**, ROC-AUC **0.6881**,
Brier **0.0945** — slightly above validation, so no sign of val overfitting.
ROC-AUC ≈ 0.68 is the published ceiling for this dataset; the win is measured
properly, not inflated.

### Why — the reasoning behind each decision
- **Class weights, not resampling**: weighting changes the loss, not the data —
  no duplicated patients to interact badly with grouped CV, and probabilities stay
  re-calibratable.
- **PR-AUC as the tuning objective**: with 11.4% positives, ROC-AUC is dominated by
  easy negatives; PR-AUC measures what the care team feels (precision of the flags).
- **The Brier disaster and the fix**: class weighting inflated probabilities so much
  that raw Brier (0.206) was *worse than always predicting 11.4%* (0.101). Isotonic
  calibration (patient-grouped CV) dropped it to 0.097 while leaving ranking intact —
  see DESIGN_NOTES entry 4. Without this, the API's "% risk" would be fiction.
- **Cost-based threshold**: flag when `p × preventable × readmission_cost >
  intervention_cost` → `0.10` with literature numbers ($15k readmission, $300
  program, 20% preventable). On test: flags 47% of discharges, catches **70% of
  readmissions** at 16.8% precision, projected net savings ≈ **$96k per 1,000
  discharges**. Sensitivity table logged to MLflow (threshold ranges 0.05–0.30 as
  assumptions vary).
- **Honest caveat**: a 47% flag rate may exceed real care-team capacity. The cost
  model says it pays, but capacity-constrained deployment would raise the threshold
  (the sensitivity table shows e.g. preventable=10% → threshold 0.20). Recorded as a
  limitation for the model card.

### How — the non-obvious mechanics
- `RandomizedSearchCV(..., cv=StratifiedGroupKFold)` needs `groups=` passed to
  `.fit`; `CalibratedClassifierCV` has no `groups=` at all, so we pass *precomputed
  fold indices* as `cv=`.
- XGBoost/LightGBM consume the `category` dtypes natively
  (`enable_categorical=True`); the logistic baseline one-hots inside its own
  pipeline — exactly the encoding contract from Stage 2.
- MLflow ≥3.13 refused the legacy `./mlruns` file store — tracking now uses
  `sqlite:///mlflow.db` (gitignored). Inspect runs with:
  `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`.
- **Test discipline**: `finalize.py` reads the test split only after the model and
  threshold are fixed; nothing tuned can see it. That's why the test numbers are
  trustworthy.

Result: **`readmission-risk` v1 registered** — calibrated XGBoost, threshold 0.10,
test PR-AUC 0.244 / ROC-AUC 0.688 / Brier 0.095.

---

*Next: Stage 4 — Package & Deploy (FastAPI /predict with risk score + top factors,
Docker, docker-compose with Prometheus + Grafana, GCP Cloud Run).*
