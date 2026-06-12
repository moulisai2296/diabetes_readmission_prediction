"""Reproducible cleaning pipeline for the diabetes readmission dataset.

Implements the six rules agreed in the EDA (notebooks/01_eda.ipynb, "Findings" cell):
1. '?' is the missing-value marker; the literal string "None" in A1Cresult /
   max_glu_serum means "test not performed" and must survive as its own category.
2. Drop weight (96.9% missing) and the single-valued examide / citoglipton.
3. Drop rows where readmission is impossible (expired/hospice discharge) and the
   3 gender = "Unknown/Invalid" rows.
4. Keep missingness as an explicit "Missing" category for medical_specialty,
   payer_code, race.
5. Binary target: readmitted == "<30".
6. patient_nbr is kept for grouped splitting only — never as a feature.

Run:  uv run python src/data/clean.py
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

log = logging.getLogger("clean")

# Discharge dispositions where a 30-day readmission is impossible (expired / hospice),
# per data_folder/IDS_mapping.csv
DEATH_HOSPICE_IDS = [11, 13, 14, 19, 20, 21]

DROP_COLS = ["weight", "examide", "citoglipton"]
MISSING_AS_CATEGORY = ["medical_specialty", "payer_code", "race"]
NOT_MEASURED_COLS = ["A1Cresult", "max_glu_serum"]
ID_CODE_COLS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]

# Mapping descriptions that all mean "we don't know" — collapsed to one category
UNKNOWN_DESCRIPTIONS = {"NULL", "Not Available", "Not Mapped", "Unknown/Invalid", ""}


def load_raw(path: Path) -> pd.DataFrame:
    """Load the raw CSV. keep_default_na=False stops pandas turning the string
    "None" (= test not performed) into NaN; only '?' and empty marks missing."""
    return pd.read_csv(path, na_values=["?", ""], keep_default_na=False, low_memory=False)


def parse_ids_mapping(path: Path) -> dict[str, dict[int, str]]:
    """IDS_mapping.csv is three stacked id→description tables separated by blank lines."""
    sections: dict[str, dict[int, str]] = {}
    current: str | None = None
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row or not row[0].strip():
                current = None
            elif current is None:
                current = row[0]
                sections[current] = {}
            else:
                sections[current][int(row[0])] = row[1]
    return sections


def clean(df: pd.DataFrame, ids_map: dict[str, dict[int, str]]) -> pd.DataFrame:
    df = df.copy()
    n0 = len(df)
    log.info("raw rows: %s", f"{n0:,}")

    # Rule 3a: impossible readmissions (expired / hospice discharge)
    mask = df["discharge_disposition_id"].isin(DEATH_HOSPICE_IDS)
    df = df[~mask]
    log.info("dropped %s expired/hospice rows -> %s", f"{mask.sum():,}", f"{len(df):,}")

    # Rule 3b: unusable gender rows
    mask = df["gender"] == "Unknown/Invalid"
    df = df[~mask]
    log.info("dropped %s Unknown/Invalid gender rows -> %s", f"{mask.sum():,}", f"{len(df):,}")

    # Rule 2: dead columns
    df = df.drop(columns=DROP_COLS)
    log.info("dropped columns: %s", ", ".join(DROP_COLS))

    # Rule 1: "None" in lab columns = test not performed, a real category
    for col in NOT_MEASURED_COLS:
        df[col] = df[col].replace("None", "NotMeasured")

    # Rule 4: missingness is informative — keep it visible
    for col in MISSING_AS_CATEGORY:
        n_miss = df[col].isna().sum()
        df[col] = df[col].fillna("Missing")
        log.info("%s: %s missing -> 'Missing' category", col, f"{n_miss:,}")

    # Map admission/discharge/source ids to descriptions, collapsing the various
    # "unknown" spellings into one category
    for col in ID_CODE_COLS:
        desc = df[col].map(ids_map[col])
        df[col.removesuffix("_id")] = desc.where(~desc.isin(UNKNOWN_DESCRIPTIONS), "Unknown").fillna("Unknown")
    df = df.drop(columns=ID_CODE_COLS)

    # Rule 5: binary target
    df["target"] = (df["readmitted"] == "<30").astype("int8")
    log.info(
        "target rate: %.4f (%s of %s)", df["target"].mean(), f"{df['target'].sum():,}", f"{len(df):,}"
    )
    log.info("kept %s of %s rows (%.2f%%)", f"{len(df):,}", f"{n0:,}", len(df) / n0 * 100)
    return df


# The 21 medication columns kept after dropping the single-valued examide/citoglipton
MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "insulin",
    "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

INT_COLS = [
    "encounter_id", "patient_nbr", "time_in_hospital", "num_lab_procedures",
    "num_procedures", "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]

STR_COLS = [
    "race", "gender", "age", "payer_code", "medical_specialty", "max_glu_serum",
    "A1Cresult", "change", "diabetesMed", "readmitted",
    "admission_type", "discharge_disposition", "admission_source",
] + MED_COLS

# Dtype-only entries for every expected column; strict=True makes the schema a full
# column contract — a missing, extra, or renamed column fails validation.
_columns: dict[str, pa.Column] = {
    **{c: pa.Column(int) for c in INT_COLS},
    **{c: pa.Column(str) for c in STR_COLS},
    **{c: pa.Column(str, nullable=True) for c in ("diag_1", "diag_2", "diag_3")},
    "target": pa.Column("int8"),
}

# Targeted value checks guarding each cleaning rule (override the dtype-only entries)
_columns.update(
    {
        "encounter_id": pa.Column(int, unique=True),
        "gender": pa.Column(str, pa.Check.isin(["Male", "Female"])),
        "race": pa.Column(str, pa.Check(lambda s: s.notna().all() and (s != "?").all())),
        "A1Cresult": pa.Column(str, pa.Check.isin(["NotMeasured", "Norm", ">7", ">8"])),
        "max_glu_serum": pa.Column(str, pa.Check.isin(["NotMeasured", "Norm", ">200", ">300"])),
        "discharge_disposition": pa.Column(
            str, pa.Check(lambda s: ~s.str.contains("Expired|Hospice", case=False), name="no_expired_hospice")
        ),
        "time_in_hospital": pa.Column(int, pa.Check.in_range(1, 14)),
        "number_inpatient": pa.Column(int, pa.Check.ge(0)),
        "target": pa.Column("int8", pa.Check.isin([0, 1])),
    }
)

SCHEMA = pa.DataFrameSchema(
    _columns,
    strict=True,
    checks=pa.Check(
        lambda d: ~d.select_dtypes("object").apply(lambda s: s.eq("?").any()).any(),
        name="no_question_marks_anywhere",
    ),
)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=root / "data_folder/diabetic_data.csv")
    ap.add_argument("--ids", type=Path, default=root / "data_folder/IDS_mapping.csv")
    ap.add_argument("--out", type=Path, default=root / "data_folder/processed/cleaned.parquet")
    args = ap.parse_args()

    df = clean(load_raw(args.raw), parse_ids_mapping(args.ids))
    SCHEMA.validate(df, lazy=True)
    log.info("schema validation passed")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    log.info("wrote %s (%s rows x %s cols)", args.out, f"{len(df):,}", df.shape[1])


if __name__ == "__main__":
    main()
