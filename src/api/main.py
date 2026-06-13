"""FastAPI service for 30-day readmission risk.

Endpoints:
  GET  /health   liveness + loaded model version
  POST /predict  calibrated risk score + flag + top contributing factors
  GET  /metrics  Prometheus exposition (latency, request/flag counts, score dist)

Serving artifacts (model.joblib, feature_spec.json) are produced by
`python -m src.api.export_model` and live in $ARTIFACTS_DIR (default ./artifacts).
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import markdown
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src.api.featurize import featurize, load_spec
from src.api.schema import Factor, PatientEncounter, PredictionResponse
from src.monitoring.drift import current_frame, drift_report, read_recent

log = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = Path(os.getenv("ARTIFACTS_DIR", ROOT / "artifacts"))
STATIC = Path(__file__).resolve().parent / "static"
GOVERNANCE = Path(os.getenv("GOVERNANCE_DIR", ROOT / "governance"))

# Every scored request is appended here (audit trail + the window /drift reads).
AUDIT_LOG = Path(os.getenv("AUDIT_LOG", ROOT / "logs" / "predictions.jsonl"))
DRIFT_WINDOW = int(os.getenv("DRIFT_WINDOW", "500"))       # recent requests compared
MIN_DRIFT_SAMPLES = int(os.getenv("MIN_DRIFT_SAMPLES", "30"))  # below this, refuse

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
        ref_path = artifacts / "drift_reference.parquet"
        self.reference = pd.read_parquet(ref_path) if ref_path.exists() else None
        AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        log.info("loaded model v%s, threshold %.3f, drift_reference=%s",
                 self.version, self.threshold, "yes" if self.reference is not None else "MISSING")

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
        encounter_dict = encounter.model_dump(by_alias=True)
        X = featurize([encounter_dict], self.spec)
        risk = float(self.model.predict_proba(X)[0, 1])
        flagged = risk >= self.threshold
        RISK.observe(risk)
        if flagged:
            FLAGGED.inc()
        log.info("prediction risk=%.4f flagged=%s model_v=%s", risk, flagged, self.version)
        self._audit(encounter_dict, risk, flagged)
        return PredictionResponse(
            readmission_risk=round(risk, 4), flagged=flagged,
            threshold=self.threshold, model_version=self.version,
            top_factors=self.top_factors(X),
        )

    def _audit(self, encounter: dict, risk: float, flagged: bool) -> None:
        """Append one line to the prediction audit log (lineage + drift window)."""
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "model_version": self.version,
            "risk": round(risk, 6),
            "flagged": flagged,
            "encounter": encounter,
        }
        try:
            with AUDIT_LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:  # auditing must never break a prediction
            log.exception("audit write failed")


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


# (key, tab label, markdown file) — the governance reports rendered at /governance.
_GOV_TABS = [
    ("model-card", "Model Card", "MODEL_CARD.md"),
    ("fairness", "Fairness Audit", "fairness_report.md"),
    ("shap", "SHAP Importance", "shap_global_importance.md"),
]


def _render_md(filename: str) -> str:
    path = GOVERNANCE / filename
    if not path.exists():
        return ("<p><em>Not generated yet — run "
                "<code>python -m src.governance.fairness_audit</code> and "
                "<code>python -m src.governance.shap_global</code>.</em></p>")
    return markdown.markdown(
        path.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "sane_lists"],
    )


@app.get("/governance", include_in_schema=False)
def governance() -> HTMLResponse:
    """Governance dashboard: model card, fairness audit, and global SHAP importance."""
    buttons, panels = [], []
    for i, (key, label, filename) in enumerate(_GOV_TABS):
        active = " active" if i == 0 else ""
        buttons.append(f'<button class="tab{active}" data-tab="{key}">{label}</button>')
        body = _render_md(filename)
        if key == "shap":
            body += ('<img src="/governance/shap_importance.png" alt="Global SHAP importance" '
                     'class="shap-img">')
        panels.append(f'<section class="panel{active}" id="{key}">{body}</section>')

    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Governance — Readmission Risk</title>
<style>
 :root {{ --bg:#d7e9dd; --card:#fff; --muted:#5a6f63; --line:#c2d6c9; --text:#16261d; --accent:#0b6b4f; }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--text); }}
 header {{ background:var(--card); border-bottom:1px solid var(--line); padding:16px 28px; }}
 header h1 {{ margin:0; font-size:19px; color:var(--accent); }}
 header a {{ color:var(--accent); font-size:13px; margin-right:14px; text-decoration:none; }}
 .tabs {{ display:flex; gap:6px; padding:14px 28px 0; flex-wrap:wrap; }}
 .tab {{ background:transparent; border:1px solid var(--line); border-bottom:none; border-radius:8px 8px 0 0;
         padding:9px 16px; font-size:14px; cursor:pointer; color:var(--muted); }}
 .tab.active {{ background:var(--card); color:var(--accent); font-weight:600; }}
 .panel {{ display:none; background:var(--card); border:1px solid var(--line); border-radius:0 12px 12px 12px;
           margin:0 28px 28px; padding:24px 28px; max-width:1000px; line-height:1.5; }}
 .panel.active {{ display:block; }}
 .panel table {{ border-collapse:collapse; width:100%; margin:12px 0; font-size:13px; }}
 .panel th, .panel td {{ border:1px solid var(--line); padding:7px 9px; text-align:left; }}
 .panel th {{ background:#eef5f0; }}
 .panel h1 {{ font-size:22px; }} .panel h2 {{ font-size:18px; margin-top:24px; }} .panel h3 {{ font-size:15px; }}
 .panel code {{ background:#eef5f0; padding:1px 5px; border-radius:4px; }}
 .shap-img {{ max-width:100%; margin-top:16px; border:1px solid var(--line); border-radius:8px; }}
</style></head><body>
<header>
  <h1>Governance — 30-Day Readmission Risk</h1>
  <nav><a href="/">← Demo</a><a href="/drift">Drift report</a><a href="/docs">API docs</a></nav>
</header>
<div class="tabs">{''.join(buttons)}</div>
{''.join(panels)}
<script>
 document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => {{
   const key = t.dataset.tab;
   document.querySelectorAll('.tab').forEach(x => x.classList.toggle('active', x === t));
   document.querySelectorAll('.panel').forEach(p => p.classList.toggle('active', p.id === key));
 }}));
</script></body></html>"""
    return HTMLResponse(page)


@app.get("/governance/shap_importance.png", include_in_schema=False)
def governance_shap_png() -> Response:
    path = GOVERNANCE / "shap_global_importance.png"
    if not path.exists():
        return Response(status_code=404)
    return FileResponse(path, media_type="image/png")


@app.get("/drift", include_in_schema=True)
def drift() -> HTMLResponse:
    """Evidently data + prediction drift: recent requests vs the training baseline.

    Rebuilds the served feature matrix from the audit log and compares it to
    `drift_reference.parquet`. Returns the full Evidently report as HTML.
    """
    def _page(msg: str, code: int = 200) -> HTMLResponse:
        return HTMLResponse(f"<h2>Drift report</h2><p>{msg}</p>", status_code=code)

    if service.reference is None:
        return _page("No drift_reference.parquet in artifacts — re-run "
                     "<code>python -m src.api.export_model</code>.", 503)

    records = read_recent(AUDIT_LOG, limit=DRIFT_WINDOW)
    if len(records) < MIN_DRIFT_SAMPLES:
        return _page(f"Only {len(records)} prediction(s) logged; need "
                     f"≥ {MIN_DRIFT_SAMPLES}. Make more /predict calls, then refresh.")
    try:
        current = current_frame(records, service.spec)
        snapshot = drift_report(service.reference, current, service.spec)
        # as_iframe=False renders the report inline; the iframe variant relies on a
        # ResizeObserver that collapses to ~0 height when served directly as a response.
        return HTMLResponse(snapshot.get_html_str(as_iframe=False))
    except Exception:
        log.exception("drift report failed")
        return _page("Failed to build the drift report — see server logs.", 500)
