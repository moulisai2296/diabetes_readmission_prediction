"""Sanity checks for the cleaning pipeline against the real raw data."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.clean import DEATH_HOSPICE_IDS, SCHEMA, clean, load_raw, parse_ids_mapping

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_folder/diabetic_data.csv"
IDS = ROOT / "data_folder/IDS_mapping.csv"

pytestmark = pytest.mark.skipif(not RAW.exists(), reason="raw data not present")


@pytest.fixture(scope="module")
def cleaned() -> pd.DataFrame:
    return clean(load_raw(RAW), parse_ids_mapping(IDS))


def test_schema_passes(cleaned):
    SCHEMA.validate(cleaned, lazy=True)


def test_not_measured_is_preserved(cleaned):
    # "None" must become NotMeasured, never NaN — a missing A1c is information
    assert (cleaned["A1Cresult"] == "NotMeasured").sum() > 80_000
    assert cleaned["A1Cresult"].notna().all()


def test_impossible_readmissions_removed(cleaned):
    raw = load_raw(RAW)
    assert raw["discharge_disposition_id"].isin(DEATH_HOSPICE_IDS).any()
    assert not cleaned["discharge_disposition"].str.contains("Expired|Hospice", case=False).any()


def test_dropped_columns_and_rows(cleaned):
    for col in ("weight", "examide", "citoglipton"):
        assert col not in cleaned.columns
    assert (cleaned["gender"] != "Unknown/Invalid").all()


def test_target_rate_plausible(cleaned):
    # ~11% before exclusions; excluding never-readmitted hospice rows nudges it up
    assert 0.10 < cleaned["target"].mean() < 0.13
