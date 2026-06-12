"""Tests for feature engineering — unit tests on the pure functions plus
sanity checks on the real pipeline output."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import SCHEMA, build_features, icd9_bucket

ROOT = Path(__file__).resolve().parents[1]
CLEANED = ROOT / "data_folder/processed/cleaned.parquet"


class TestIcd9Bucket:
    @pytest.mark.parametrize(
        ("code", "bucket"),
        [
            ("428", "Circulatory"),       # heart failure — top code in the data
            ("785", "Circulatory"),
            ("785.6", "Circulatory"),
            ("486", "Respiratory"),       # pneumonia
            ("787.01", "Digestive"),
            ("250.83", "Diabetes"),
            ("250", "Diabetes"),
            ("999", "Injury"),
            ("715", "Musculoskeletal"),
            ("599.0", "Genitourinary"),
            ("197", "Neoplasms"),
            ("V57", "Other"),             # supplementary codes
            ("E909", "Other"),
            ("3", "Other"),               # infectious — not in any named bucket
            (np.nan, "Missing"),
        ],
    )
    def test_buckets(self, code, bucket):
        assert icd9_bucket(code) == bucket


@pytest.mark.skipif(not CLEANED.exists(), reason="cleaned data not present (run dvc repro)")
class TestRealPipeline:
    @pytest.fixture(scope="class")
    def features(self) -> pd.DataFrame:
        return build_features(pd.read_parquet(CLEANED))

    def test_schema_passes(self, features):
        SCHEMA.validate(features, lazy=True)

    def test_no_leakage_columns(self, features):
        assert "readmitted" not in features.columns

    def test_engineered_features_plausible(self, features):
        # most encounters change 0 or 1 meds; insulin alone is non-No in ~half
        assert features["n_med_changes"].between(0, 21).all()
        assert features["n_active_meds"].median() >= 1
        # circulatory should be the biggest primary-diagnosis bucket (EDA: codes 428/414 on top)
        assert features["diag_1_group"].value_counts().idxmax() == "Circulatory"

    def test_rare_lumping(self, features):
        for col in ("medical_specialty", "payer_code"):
            shares = features[col].value_counts(normalize=True)
            # every surviving category is either >=1% or the lump bucket itself
            small = shares[shares < 0.01].index.tolist()
            assert small in ([], ["Other"]) or set(small) <= {"Other"}

    def test_age_ordinal_replaces_age(self, features):
        assert "age" not in features.columns
        assert features["age_ordinal"].between(0, 9).all()
