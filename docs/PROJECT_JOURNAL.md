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

*Next: Stage 3 — Modeling & Evaluation (patient-grouped splits, logistic baseline,
XGBoost vs LightGBM in MLflow, PR-AUC / recall-at-precision / calibration, threshold
choice).*
