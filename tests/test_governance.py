"""Smoke tests for the governance analyses (skip if artifacts not built)."""

from pathlib import Path

import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
TEST_SPLIT = ROOT / "data_folder/processed/splits/test.parquet"

pytestmark = pytest.mark.skipif(
    not ((ARTIFACTS / "model.joblib").exists() and TEST_SPLIT.exists()),
    reason="artifacts or test split not built",
)


@pytest.fixture(scope="module")
def setup():
    from src.api.featurize import load_spec
    spec = load_spec(ARTIFACTS / "feature_spec.json")
    model = joblib.load(ARTIFACTS / "model.joblib")
    df = pd.read_parquet(TEST_SPLIT).sample(500, random_state=0)
    return spec, model, df


def test_fairness_audit_feature(setup):
    from src.governance.fairness_audit import _predict, audit_feature
    spec, model, df = setup
    y_true = df["target"].astype(int).to_numpy()
    y_pred = _predict(model, spec, df)
    res = audit_feature(y_true, y_pred, df["gender"].astype(str), "gender")
    assert set(res["by_group"]) == {"Female", "Male"}
    for row in res["by_group"].values():
        assert 0.0 <= row["recall"] <= 1.0
        assert 0.0 <= row["selection_rate"] <= 1.0
    assert res["max_difference"]["recall"] >= 0.0


def test_global_importance_ranks_features(setup):
    from src.governance.shap_global import global_importance
    spec, model, df = setup
    imp = global_importance(model, df[spec["feature_order"]])
    assert list(imp.index) == list(imp.sort_values(ascending=False).index)  # sorted desc
    assert (imp >= 0).all() and imp.sum() > 0
    assert len(imp) == len(spec["feature_order"])
