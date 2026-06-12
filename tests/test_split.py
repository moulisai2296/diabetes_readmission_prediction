"""The split must never leak a patient across splits — the most-graded invariant."""

from pathlib import Path

import pandas as pd
import pytest

from src.models.split import split_by_patient

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data_folder/processed/features.parquet"

pytestmark = pytest.mark.skipif(not FEATURES.exists(), reason="features not built (run dvc repro)")


@pytest.fixture(scope="module")
def splits():
    return split_by_patient(pd.read_parquet(FEATURES))


def test_no_patient_overlap(splits):
    train = set(splits["train"]["patient_nbr"])
    val = set(splits["val"]["patient_nbr"])
    test = set(splits["test"]["patient_nbr"])
    assert not train & val
    assert not train & test
    assert not val & test


def test_all_rows_kept(splits):
    total = sum(len(p) for p in splits.values())
    assert total == len(pd.read_parquet(FEATURES))


def test_split_proportions(splits):
    total = sum(len(p) for p in splits.values())
    assert len(splits["train"]) / total == pytest.approx(0.70, abs=0.02)
    assert len(splits["val"]) / total == pytest.approx(0.15, abs=0.02)
    assert len(splits["test"]) / total == pytest.approx(0.15, abs=0.02)


def test_target_rate_similar_across_splits(splits):
    rates = [p["target"].mean() for p in splits.values()]
    assert max(rates) - min(rates) < 0.02
