"""Feature engineering for the diabetes readmission dataset.

Consumes the cleaned parquet from src/data/clean.py and produces the model-ready
feature table. Every feature and its rationale is documented in
src/features/FEATURES.md — keep that file in sync with this one.

Design decisions (agreed during Stage 2 discussion):
- ICD-9 diagnoses are grouped into the clinical buckets from Strack et al. 2014,
  the original paper on this dataset.
- Rare categories of medical_specialty / payer_code (<1% of rows) are lumped into
  "Other". Frequency-based lumping uses no target information, so it is safe to do
  before the train/test split.
- NO model-specific encoding happens here: categoricals are emitted as pandas
  `category` dtype, and each model pipeline encodes them as it needs (one-hot for
  logistic regression, native categorical for LightGBM/XGBoost).

Run:  uv run python src/features/build_features.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from src.data.clean import MED_COLS

log = logging.getLogger("features")

RARE_LUMP_COLS = ["medical_specialty", "payer_code"]
RARE_THRESHOLD = 0.01  # categories under 1% of rows are lumped into "Other"

DIAG_BUCKETS = [
    "Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury",
    "Musculoskeletal", "Genitourinary", "Neoplasms", "Other", "Missing",
]

# age brackets are ordered — encode the order instead of one-hotting it away
AGE_ORDER = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
             "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]

CATEGORICAL_COLS = [
    "race", "gender", "admission_type", "discharge_disposition", "admission_source",
    "medical_specialty", "payer_code", "max_glu_serum", "A1Cresult",
    "change", "diabetesMed", "diag_1_group", "diag_2_group", "diag_3_group",
] + MED_COLS


def icd9_bucket(code: str | float) -> str:
    """Map a raw ICD-9 code (e.g. '428', '250.83', 'V57') to a clinical bucket.

    Grouping follows Strack et al. 2014 (the original study on this dataset).
    """
    if pd.isna(code):
        return "Missing"
    code = str(code)
    if code.startswith(("V", "E")):
        return "Other"  # supplementary / external-cause codes
    num = float(code)
    if 390 <= num <= 459 or int(num) == 785:
        return "Circulatory"
    if 460 <= num <= 519 or int(num) == 786:
        return "Respiratory"
    if 520 <= num <= 579 or int(num) == 787:
        return "Digestive"
    if int(num) == 250:
        return "Diabetes"
    if 800 <= num <= 999:
        return "Injury"
    if 710 <= num <= 739:
        return "Musculoskeletal"
    if 580 <= num <= 629 or int(num) == 788:
        return "Genitourinary"
    if 140 <= num <= 239:
        return "Neoplasms"
    return "Other"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- diagnoses: 716-789 raw ICD-9 codes -> 10 clinical buckets ---
    for col in ("diag_1", "diag_2", "diag_3"):
        mapping = {c: icd9_bucket(c) for c in df[col].dropna().unique()}
        df[f"{col}_group"] = df[col].map(mapping).fillna("Missing")
    df["diabetes_diag_any"] = (
        (df[["diag_1_group", "diag_2_group", "diag_3_group"]] == "Diabetes").any(axis=1).astype("int8")
    )
    df = df.drop(columns=["diag_1", "diag_2", "diag_3"])
    log.info("diag_1 bucket shares:\n%s",
             df["diag_1_group"].value_counts(normalize=True).round(3).to_string())

    # --- utilization: prior-visit pressure (number_inpatient is the strongest
    # single signal in the EDA; components are kept alongside the total) ---
    df["total_prior_visits"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )

    # --- medication regimen: how much was the treatment adjusted? ---
    meds = df[MED_COLS]
    df["n_med_changes"] = meds.isin(["Up", "Down"]).sum(axis=1).astype("int16")
    df["n_active_meds"] = (meds != "No").sum(axis=1).astype("int16")

    # --- age: ordered brackets -> ordinal 0..9 ---
    df["age_ordinal"] = df["age"].map({b: i for i, b in enumerate(AGE_ORDER)}).astype("int8")
    df = df.drop(columns=["age"])

    # --- rare-category lumping for the two high-cardinality context columns ---
    for col in RARE_LUMP_COLS:
        shares = df[col].value_counts(normalize=True)
        rare = shares[shares < RARE_THRESHOLD].index
        df[col] = df[col].where(~df[col].isin(rare), "Other")
        log.info("%s: lumped %d rare categories -> %d remain",
                 col, len(rare), df[col].nunique())

    # readmitted (3-class source of `target`) must never reach a model
    df = df.drop(columns=["readmitted"])

    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")

    log.info("feature table: %s rows x %s cols", f"{len(df):,}", df.shape[1])
    return df


SCHEMA = pa.DataFrameSchema(
    {
        **{f"diag_{i}_group": pa.Column("category", pa.Check.isin(DIAG_BUCKETS)) for i in (1, 2, 3)},
        "age_ordinal": pa.Column("int8", pa.Check.in_range(0, 9)),
        "n_med_changes": pa.Column("int16", pa.Check.in_range(0, len(MED_COLS))),
        "n_active_meds": pa.Column("int16", pa.Check.in_range(0, len(MED_COLS))),
        "total_prior_visits": pa.Column(int, pa.Check.ge(0)),
        "diabetes_diag_any": pa.Column("int8", pa.Check.isin([0, 1])),
        "target": pa.Column("int8", pa.Check.isin([0, 1])),
        "patient_nbr": pa.Column(int),
    },
    checks=[
        pa.Check(lambda d: "readmitted" not in d.columns, name="raw_label_removed"),
        pa.Check(
            lambda d: (d["total_prior_visits"]
                       == d.number_outpatient + d.number_emergency + d.number_inpatient).all(),
            name="total_prior_visits_consistent",
        ),
    ],
)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path,
                    default=root / "data_folder/processed/cleaned.parquet")
    ap.add_argument("--out", type=Path,
                    default=root / "data_folder/processed/features.parquet")
    args = ap.parse_args()

    df = build_features(pd.read_parquet(args.inp))
    SCHEMA.validate(df, lazy=True)
    log.info("schema validation passed")

    df.to_parquet(args.out, index=False)
    log.info("wrote %s (%s rows x %s cols)", args.out, f"{len(df):,}", df.shape[1])


if __name__ == "__main__":
    main()
