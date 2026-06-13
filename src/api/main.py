"""FastAPI service for 30-day readmission risk.

Endpoints:
  GET  /health   liveness + loaded model version
  POST /predict  calibrated risk score + flag + top contributing factors
  GET  /metrics  Prometheus exposition (latency, request/flag counts, score dist)

Serving artifacts (model.joblib, feature_spec.json) are produced by
`python -m src.api.export_model` and live in $ARTIFACTS_DIR (default ./artifacts).
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src.api.featurize import featurize, load_spec
from src.api.schema import Factor, PatientEncounter, PredictionResponse

log = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = Path(os.getenv("ARTIFACTS_DIR", ROOT / "artifacts"))
STATIC = Path(__file__).resolve().parent / "static"

REQUESTS = Counter("predict_requests_total", "Prediction requests")
ERRORS = Counter("predict_errors_total", "Prediction errors")
FLAGGED = Counter("predict_flagged_total", "Patients flagged for follow-up")
LATENCY = Histogram("predict_latency_seconds", "Prediction latency")
RISK = Histogram("predict_risk_score", "Predicted risk distribution",
                 buckets=(0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0))


def _base_estimators(model) -> list:
    """The fitted XGBoost behind each calibration fold (for SHAP contributions)."""
    out = []
    for cc in getattr(model, "calibrated_classifiers_", []):
        est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
        if est is not None:
            out.append(est)
    return out


class Service:
    def __init__(self, artifacts: Path):
        self.spec = load_spec(artifacts / "feature_spec.json")
        self.model = joblib.load(artifacts / "model.joblib")
        self.threshold = float(self.spec["threshold"])
        self.version = str(self.spec.get("model_version", "unknown"))
        self.boosters = [e.get_booster() for e in _base_estimators(self.model)]
        log.info("loaded model v%s, threshold %.3f", self.version, self.threshold)

    def top_factors(self, X: pd.DataFrame, k: int = 5) -> list[Factor]:
        """Average XGBoost SHAP contributions across calibration folds."""
        if not self.boosters:
            return []
        try:
            dm = xgb.DMatrix(X, enable_categorical=True)
            contribs = np.mean([b.predict(dm, pred_contribs=True) for b in self.boosters], axis=0)
            row = contribs[0][:-1]  # drop bias term
            order = np.argsort(np.abs(row))[::-1][:k]
            factors = []
            for i in order:
                name = self.spec["feature_order"][i]
                factors.append(Factor(
                    feature=name, value=str(X.iloc[0, i]),
                    contribution=round(float(row[i]), 4),
                    direction="increases" if row[i] > 0 else "decreases",
                ))
            return factors
        except Exception:  # explanations are best-effort; never fail a prediction
            log.exception("top_factors failed")
            return []

    def predict(self, encounter: PatientEncounter) -> PredictionResponse:
        X = featurize([encounter.model_dump(by_alias=True)], self.spec)
        risk = float(self.model.predict_proba(X)[0, 1])
        flagged = risk >= self.threshold
        RISK.observe(risk)
        if flagged:
            FLAGGED.inc()
        log.info("prediction risk=%.4f flagged=%s model_v=%s", risk, flagged, self.version)
        return PredictionResponse(
            readmission_risk=round(risk, 4), flagged=flagged,
            threshold=self.threshold, model_version=self.version,
            top_factors=self.top_factors(X),
        )


service: Service | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    service = Service(ARTIFACTS)
    yield


app = FastAPI(title="Diabetes 30-Day Readmission Risk", version="1.0", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Minimal demo UI: form -> /predict -> risk + top factors."""
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict:
    ready = service is not None
    return {"status": "ok" if ready else "loading",
            "model_version": service.version if ready else None}


@app.post("/predict", response_model=PredictionResponse)
def predict(encounter: PatientEncounter) -> PredictionResponse:
    REQUESTS.inc()
    with LATENCY.time():
        try:
            return service.predict(encounter)
        except Exception:
            ERRORS.inc()
            raise


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
