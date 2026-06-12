# Feature Log

Every feature in `data_folder/processed/features.parquet`, where it comes from, and why
it exists. Required by the project brief ("keep a written record of every feature and why
it exists"). Built by `src/features/build_features.py` from the cleaned data.

## Identifiers (never used as model features)

| Column | Why it's kept |
|---|---|
| `encounter_id` | joins predictions back to source rows; audit trail |
| `patient_nbr` | **grouped train/val/test splitting only** — 23.5% of patients have >1 encounter (46% of rows); a row-level split would leak |

## Target

| Column | Definition | Why |
|---|---|---|
| `target` | 1 if `readmitted == "<30"` else 0 | the business question: who needs follow-up within 30 days. The raw 3-class `readmitted` column is **dropped** so it can never leak into a model |

## Numerical — passed through from source

| Feature | Why it exists |
|---|---|
| `time_in_hospital` | longer stays correlate with sicker patients (EDA corr 0.044) |
| `num_lab_procedures` | care intensity proxy |
| `num_procedures` | care intensity proxy |
| `num_medications` | regimen complexity (corr 0.038) |
| `number_outpatient` | prior-year outpatient visits — utilization history |
| `number_emergency` | prior-year ER visits — 2nd-strongest univariate signal (corr 0.061) |
| `number_inpatient` | prior-year inpatient visits — **strongest signal in the EDA**: readmission rate climbs 8.4% → 36.4% from 0 to 5+ visits (corr 0.165) |
| `number_diagnoses` | comorbidity burden (corr 0.050) |

## Numerical — engineered

| Feature | Definition | Why it exists |
|---|---|---|
| `total_prior_visits` | outpatient + emergency + inpatient | overall "system contact" pressure in one number; components kept so the model can still weight inpatient higher |
| `n_med_changes` | count of the 21 med columns set to `Up`/`Down` | a dose changed at discharge = unstable regimen; per the brief, medication change count is a known signal |
| `n_active_meds` | count of the 21 med columns ≠ `No` | polypharmacy burden, complements `num_medications` (which counts all meds, not just diabetes meds) |
| `age_ordinal` | bracket index 0–9 (`[0-10)`→0 … `[90-100)`→9) | age brackets are *ordered*; one-hot encoding would discard the ordering. Replaces raw `age` |
| `diabetes_diag_any` | 1 if any of the 3 diagnoses is ICD-9 250.xx | distinguishes "admitted *for* diabetes" from "diabetic, admitted for something else" |

## Categorical — diagnoses (engineered)

| Feature | Definition | Why it exists |
|---|---|---|
| `diag_1_group`, `diag_2_group`, `diag_3_group` | raw ICD-9 code (716–789 uniques) → 10 clinical buckets: Circulatory, Respiratory, Digestive, Diabetes, Injury, Musculoskeletal, Genitourinary, Neoplasms, Other, Missing | 700+ categories would overfit / explode one-hot width; buckets follow **Strack et al. 2014**, the original study on this dataset, so the grouping is citable, clinically meaningful, and reproducible. Raw `diag_*` are dropped |

## Categorical — passed through (as pandas `category` dtype)

| Feature | Notes |
|---|---|
| `race` | includes explicit `Missing` category (2.2%); needed for the fairness audit |
| `gender` | Male/Female only after cleaning; fairness-audit dimension |
| `admission_type` | mapped from id via IDS_mapping; NULL/Not Available/Not Mapped collapsed to `Unknown` |
| `discharge_disposition` | as above; expired/hospice rows already excluded |
| `admission_source` | as above |
| `medical_specialty` | 49% `Missing` (informative); categories <1% of rows lumped into `Other` |
| `payer_code` | 40% `Missing`; same <1% lumping |
| `max_glu_serum` | `NotMeasured` / Norm / >200 / >300 — "not measured" is signal (EDA: unmeasured patients readmit more) |
| `A1Cresult` | `NotMeasured` / Norm / >7 / >8 — same trap, same handling |
| `change` | any diabetes med changed during stay (Ch/No) |
| `diabetesMed` | any diabetes med prescribed (Yes/No) |
| 21 individual med columns (`metformin` … `metformin-pioglitazone`) | No/Steady/Up/Down each. Kept individually: near-constant but tree models handle them cheaply, and SHAP will tell us later if they earn their place — pruning before modeling would be unjustified |

## Deliberately NOT included

| Dropped | Why |
|---|---|
| `readmitted` | 3-class source of the target — leakage by definition |
| `age` (raw brackets) | replaced by `age_ordinal` |
| `weight`, `examide`, `citoglipton` | dropped in cleaning (97% missing / single-valued) |
| `diag_1/2/3` raw codes | replaced by `*_group` buckets |

## Encoding contract

This table contains **no model-specific encoding**. Categoricals are `category` dtype;
each model pipeline encodes them itself (one-hot for the logistic baseline, native
categorical splits for LightGBM/XGBoost). Rationale: different models want different
encodings, and the serving API later receives raw-ish values and lets the pipeline
transform them — one source of truth for the transformation.
