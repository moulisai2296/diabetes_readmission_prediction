"""API tests against the real exported model (skip if artifacts not built)."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS / "model.joblib").exists(),
    reason="serving artifacts not built (run: python -m src.api.export_model)",
)


@pytest.fixture(scope="module")
def client():
    from src.api.main import app
    with TestClient(app) as c:  # triggers startup -> model load
        yield c


@pytest.fixture
def payload():
    from src.api.schema import EXAMPLE_ENCOUNTER
    return dict(EXAMPLE_ENCOUNTER)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_version"] == "1"


def test_predict_shape(client, payload):
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["readmission_risk"] <= 1.0
    assert body["flagged"] == (body["readmission_risk"] >= body["threshold"])
    assert len(body["top_factors"]) == 5
    assert all(f["direction"] in ("increases", "decreases") for f in body["top_factors"])


def test_high_risk_patient_flagged(client, payload):
    # many prior inpatient visits -> the strongest signal -> should clear threshold 0.10
    payload["number_inpatient"] = 8
    payload["number_emergency"] = 4
    body = client.post("/predict", json=payload).json()
    assert body["flagged"] is True


def test_validation_rejects_bad_age(client, payload):
    payload["age"] = "seventy"
    assert client.post("/predict", json=payload).status_code == 422


def test_defaults_allow_minimal_payload(client):
    minimal = {"gender": "Male", "age": "[60-70)", "time_in_hospital": 3,
               "num_lab_procedures": 40, "num_procedures": 1, "num_medications": 10,
               "number_outpatient": 0, "number_emergency": 0, "number_inpatient": 0,
               "number_diagnoses": 6, "diag_1": "250.83"}
    assert client.post("/predict", json=minimal).status_code == 200


def test_metrics_endpoint(client, payload):
    client.post("/predict", json=payload)
    text = client.get("/metrics").text
    assert "predict_requests_total" in text
    assert "predict_risk_score" in text


def test_governance_dashboard(client):
    r = client.get("/governance")
    assert r.status_code == 200
    assert all(t in r.text for t in ("Model Card", "Fairness Audit", "SHAP Importance"))
    png = client.get("/governance/shap_importance.png")
    assert png.status_code == 200
    assert png.headers["content-type"] == "image/png"


def test_drift_needs_minimum(client, monkeypatch, tmp_path):
    import src.api.main as main
    monkeypatch.setattr(main, "AUDIT_LOG", tmp_path / "empty.jsonl")
    monkeypatch.setattr(main, "MIN_DRIFT_SAMPLES", 5)
    r = client.get("/drift")
    assert r.status_code == 200
    assert "need" in r.text.lower()  # the "not enough data" page


def test_drift_report_renders(client, payload, monkeypatch, tmp_path):
    import src.api.main as main
    monkeypatch.setattr(main, "AUDIT_LOG", tmp_path / "preds.jsonl")
    monkeypatch.setattr(main, "MIN_DRIFT_SAMPLES", 5)
    for n in range(8):  # _audit appends each to the patched log
        client.post("/predict", json={**payload, "num_medications": 5 + n})
    r = client.get("/drift")
    assert r.status_code == 200
    # a real report, rendered inline as a full page (not the self-collapsing iframe variant)
    assert "<!doctype html" in r.text.lower()
    assert "drift" in r.text.lower()
    assert "evidently-ui-iframe" not in r.text
