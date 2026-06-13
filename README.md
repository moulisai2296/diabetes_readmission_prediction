# Diabetes 30-Day Readmission Prediction

End-to-end ML system predicting each diabetic patient's risk of readmission within
30 days of discharge, so care teams can target follow-up — from raw clinical data to a
deployed, monitored, governed service.

Dataset: UCI Diabetes 130-US Hospitals (1999–2008), ~101k encounters.

> Work in progress — built stage by stage (data engineering → features → modeling →
> deployment → observability → governance). See `CLAUDE.md` for the full plan and
> `docs/DESIGN_NOTES.md` for concept write-ups.

## Quickstart

```powershell
# Requires: uv (https://docs.astral.sh/uv/), Python 3.12 (uv installs it)
uv sync                          # create env + install pinned deps
uv run dvc repro                 # rebuild data pipeline (raw -> cleaned -> features -> splits)
uv run pytest                    # run tests

# Train, finalize (calibrate + register), and export serving artifacts
uv run python -m src.models.train
uv run python -m src.models.finalize
uv run python -m src.api.export_model      # writes artifacts/model.joblib + feature_spec.json

# Run the API, then open http://127.0.0.1:8000/docs
uv run uvicorn src.api.main:app --port 8000
```

`POST /predict` returns a calibrated 30-day readmission risk, a follow-up flag, and the
top contributing factors. `GET /health` and `GET /metrics` (Prometheus) are also exposed.

The raw data files are DVC-tracked (only `.dvc` pointers live in git). Place
`diabetic_data.csv` and `IDS_mapping.csv` in `data_folder/` if you don't have access
to a DVC remote.
