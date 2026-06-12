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
uv sync                      # create env + install pinned deps
uv run dvc repro             # rebuild the data pipeline (raw -> cleaned parquet)
uv run pytest                # run tests
```

The raw data files are DVC-tracked (only `.dvc` pointers live in git). Place
`diabetic_data.csv` and `IDS_mapping.csv` in `data_folder/` if you don't have access
to a DVC remote.
