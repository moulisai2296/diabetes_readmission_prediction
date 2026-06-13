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

## Stage 4 — Package & Deploy (part 1: the FastAPI service)

### What we did
1. **Refactored `build_features.py`** to extract `engineer()` (deterministic, row-wise
   feature math) and `lump_rare()` (the one fitted transform) — so training and serving
   share *one* engineering function. Verified the refactor left `features.parquet`
   byte-identical (downstream DVC stages didn't re-run).
2. **`src/api/export_model.py`** — loads `readmission-risk` v1 from the MLflow registry
   and writes a self-contained `artifacts/` folder: `model.joblib` +
   `feature_spec.json` (feature order, exact category vocabularies, threshold). The API
   depends on these files, not on MLflow at runtime.
3. **`src/api/featurize.py`** — serving transform: `engineer()` → apply the saved
   rare-category vocabulary → set pandas `category` dtypes with the exact training
   categories → order columns.
4. **`src/api/schema.py`** — Pydantic request (`PatientEncounter`, ~45 post-ETL fields
   with sensible defaults + validation) and response (`risk`, `flagged`, `threshold`,
   `model_version`, `top_factors`).
5. **`src/api/main.py`** — FastAPI app: `GET /health`, `POST /predict`, `GET /metrics`
   (Prometheus). `/predict` returns the calibrated risk, the flag decision (risk ≥
   0.10), and the top-5 contributing factors via XGBoost SHAP contributions.
6. **6 API tests** (35 total) and a real over-HTTP smoke test.

### The proof it works (live server, real HTTP)
Example 70-something patient, 2 prior inpatient visits, transferred to another
facility → **31.4% risk, flagged=true**, with honest explanations:

| factor | contribution | direction |
|---|---|---|
| discharge_disposition = "...another inpatient institution" | +0.59 | increases |
| number_inpatient = 2 | +0.23 | increases |
| payer_code = Missing | +0.11 | increases |
| total_prior_visits = 2 | +0.09 | increases |
| admission_source = "Transfer from a hospital" | −0.09 | decreases |

`number_inpatient` surfacing as the #2 driver matches the EDA — the model learned the
signal we expected, and SHAP confirms it per-patient.

### Why — the key design decisions
- **Server owns feature engineering; client sends raw-ish facts.** The caller posts the
  three visit counts, raw ICD-9 codes, med statuses, etc.; the server computes
  `total_prior_visits`, the ICD-9 buckets, `age_ordinal`. This is the training/serving
  skew defense — see DESIGN_NOTES entry 5. If the client computed engineered features,
  any tiny difference in logic would corrupt predictions silently.
- **Export decouples serving from MLflow.** The container shouldn't carry the sqlite db
  + `mlruns/`. We snapshot the model to joblib and the fitted vocabulary to JSON, as a
  matched pair, so the service loads two small files.
- **Calibrated probability is the headline number.** `/predict` returns the *calibrated*
  risk (honest %), and the flag uses the cost-based 0.10 threshold from Stage 3.
- **Explanations are best-effort.** `top_factors` is wrapped in try/except — an
  explanation failure logs but never fails the prediction itself.

### How — the non-obvious mechanics
- **Category codes are positional.** XGBoost native-categorical maps each category to an
  integer by its position in the dtype's category list. Serving *must* set the identical
  category order, or "Cardiology" becomes a different code than in training. We freeze
  the order in `feature_spec.json` and replay it with
  `pd.Categorical(values, categories=spec_list)`.
- **Rare-category lumping is stateful.** It keeps categories that were ≥1% *of training*
  — meaningless to recompute from one request — so the surviving set is persisted and
  applied at serving (`lump_rare(s, keep=...)`).
- **SHAP at serving** uses the underlying XGBoost boosters
  (`booster.predict(DMatrix(..., enable_categorical=True), pred_contribs=True)`),
  averaged across the 3 calibration folds; the last column (bias) is dropped, the rest
  ranked by |contribution|.
- **Hyphenated med columns** (`glyburide-metformin`, …) aren't valid Python identifiers,
  so the Pydantic model uses `Field(alias=...)` and we dump with `by_alias=True` to
  rebuild the original column names for `engineer()`.

Reproduce the serving chain after training: `dvc repro` → `python -m src.models.train`
→ `python -m src.models.finalize` → `python -m src.api.export_model` → `uvicorn
src.api.main:app`.

*Next: Stage 4 part 2 — Dockerfile, docker-compose with Prometheus + Grafana, then
GCP Cloud Run (needs Docker Desktop, installing).*

## Stage 4 — Package & Deploy (part 2: containers + local monitoring stack)

### What we did
1. **`Dockerfile`** — multi-step build on `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`:
   copy `pyproject.toml` + `uv.lock` first and `uv sync --frozen --no-install-project
   --no-dev` (cached unless the lockfile changes), *then* copy `src/` and `artifacts/`.
   `HEALTHCHECK` hits the app's own `/health`; entrypoint is uvicorn on `:8000`.
2. **`.dockerignore`** — keeps the build context small (drops `.git`, `.venv`, `mlruns`,
   `data_folder`, notebooks, tests, docs) but **deliberately keeps `artifacts/`** — the
   image needs the model + feature spec.
3. **`docker-compose.yml`** — three services: `api` (built locally), `prometheus`
   (scrapes the API), `grafana` (auto-provisioned datasource + dashboard). All
   `restart: unless-stopped`.
4. **`monitoring/`** — `prometheus.yml` scrape config, Grafana provisioning for the
   Prometheus datasource and a dashboard file, and the **Readmission Risk API**
   dashboard (request rate, p95 latency, flag rate, mean predicted risk).
5. **Brought the whole stack up on one machine and verified it end to end.**

### The proof it works (whole stack, real HTTP)
- `docker compose up -d` → all three containers `Up`; API reports `health: healthy`.
- `POST /predict` (example encounter) → **31.4% risk, flagged=true** with SHAP factors —
  identical to the bare-uvicorn run, confirming the container carries a correct model.
- Prometheus target `readmission-api` → **up**; the dashboard's exact PromQL resolves
  against live data (`predict_requests_total`=31, p95 latency ≈ 0.094 s).
- Grafana: Prometheus datasource provisioned as default, dashboard
  `uid=readmission-api` loaded.

### Why — the key design decisions
- **Snapshot artifacts into the image, not MLflow.** Same decision as part 1, now paying
  off: the container is self-contained (model + spec), no sqlite/`mlruns` baggage, no
  registry call at runtime.
- **Provision Grafana as code.** Datasource + dashboard live in `monitoring/` and are
  mounted in — the dashboard exists the moment the stack boots, so "clone → one command
  → working dashboard" (the definition of done) holds with no manual clicking.
- **Server-side feature engineering pays its second dividend.** Because the image owns
  `engineer()`, the only things that vary between bare-metal and container runs are
  irrelevant to predictions — the 31.4% matched exactly.

### How — the non-obvious mechanics (the Docker Hub pull saga)
The build worked first try (base from GHCR + deps from PyPI), but `docker compose up`
**could not pull `prom/prometheus` or `grafana/grafana`** — every attempt died with
`httpReadSeeker ... production.cloudfront.docker.com ...: EOF`. The debugging chain:
- **Not the internet / not size.** `curl`/`wget` downloaded the *exact* signed CloudFront
  blob URLs fine — including the largest 58 MB Prometheus layer, over both HTTP/1.1 and
  HTTP/2, from both the Ubuntu WSL distro and Docker's own `docker-desktop` VM.
- **Not concurrency, not the engine VM's network.** `wsl --terminate docker-desktop` and
  `max-concurrent-downloads: 1` both failed to help.
- **Isolated to one path.** `docker pull` from **GHCR worked**; `docker pull` from
  **Docker Hub failed** — the only difference being Docker Hub redirects blob data to AWS
  CloudFront, and something on this network resets *containerd's* connections to that one
  CDN (while curl/wget survive it).
- **Fix:** add `"registry-mirrors": ["https://mirror.gcr.io"]` to the Docker Engine
  config. Google's pull-through cache serves the same images off non-AWS infra; both
  images then pulled on the first try and the stack came up.

Reproduce: `python -m src.api.export_model` (writes `artifacts/`) → `docker compose up
--build` → API `http://localhost:8000/docs`, Prometheus `:9090`, Grafana `:3000`
(admin/admin). If pulls fail with the CloudFront EOF, the `registry-mirrors` line above
is the fix.

*Next: Stage 4 part 3 — GCP Cloud Run (push image to Artifact Registry via Cloud Build,
deploy, billing alert, rollback plan).*
