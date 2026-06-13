# Diabetes 30-Day Readmission Prediction

End-to-end ML system predicting each diabetic patient's risk of readmission within
30 days of discharge, so care teams can target follow-up — from raw clinical data to a
deployed, monitored, governed service.

Dataset: UCI Diabetes 130-US Hospitals (1999–2008), ~101k encounters.

> Built stage by stage (data engineering → features → modeling → deployment →
> observability → governance); GCP Cloud Run deploy is the remaining step.

## Documentation

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — start here: end-to-end data flow and
  architecture (offline training plane vs online serving plane, prediction path,
  monitoring inputs, governance timing) with rendered diagrams and an artifact map.
- [docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md) — the *why* behind key design decisions.
- [docs/PROJECT_JOURNAL.md](docs/PROJECT_JOURNAL.md) — chronological build log, stage by stage.
- [governance/MODEL_CARD.md](governance/MODEL_CARD.md) · [governance/REFLECTION.md](governance/REFLECTION.md) — model card and retrospective.
- [monitoring/README.md](monitoring/README.md) — metrics, drift, alerts, retraining trigger.
- `CLAUDE.md` — the full stage-by-stage plan and project decisions.

Once running, the service also serves these live: `/` (demo UI), `/governance`
(model card + fairness + SHAP), and `/drift` (Evidently report).

## Quickstart

### 1. Build the model and serving artifacts

```powershell
# Requires: uv (https://docs.astral.sh/uv/), Python 3.12 (uv installs it)
uv sync                          # create env + install pinned deps
uv run dvc repro                 # rebuild data pipeline (raw -> cleaned -> features -> splits)
uv run pytest                    # run tests

# Train, finalize (calibrate + register), and export serving artifacts
uv run python -m src.models.train
uv run python -m src.models.finalize
uv run python -m src.api.export_model            # writes artifacts/ (model + feature_spec + drift reference)
uv run python -m src.governance.fairness_audit   # governance gate (fairness report)
uv run python -m src.governance.shap_global      # global SHAP importance
```

### 2a. Run the full stack with Docker (recommended)

The image bundles the `artifacts/` from step 1, so export the model first, then:

```powershell
docker compose up -d --build
#   API + demo UI   http://localhost:8000/
#   Prometheus      http://localhost:9090
#   Grafana         http://localhost:3000   (admin/admin)
```

This is the whole system on one machine — API, metrics scraping, and dashboards. See
[monitoring/README.md](monitoring/README.md) for the observability walkthrough.

### 2b. Or run just the API (lightweight dev)

```powershell
uv run uvicorn src.api.main:app --port 8000     # API only — no Prometheus/Grafana
```

Either way, `POST /predict` returns a calibrated 30-day readmission risk, a follow-up
flag, and the top contributing factors; `GET /health` and `GET /metrics` (Prometheus) are
also exposed.

The raw data files are DVC-tracked (only `.dvc` pointers live in git). Place
`diabetic_data.csv` and `IDS_mapping.csv` in `data_folder/` if you don't have access
to a DVC remote.
