"""Live data + prediction drift via Evidently.

Compares the recent requests the API has actually served against the *training*
distribution the model learned from. Two things drift:

  - **Data drift** — the input feature distributions move away from training
    (e.g. an older population, more emergency admissions).
  - **Prediction drift** — the distribution of predicted risk scores moves; we
    expose this by treating the model's score as an extra numeric column.

The training reference is `artifacts/drift_reference.parquet` (a sample of the
training feature matrix plus the model's score on it), written by
`src.api.export_model`. The current window is rebuilt from the prediction audit
log by re-running the same `featurize()` the model is served with, so reference
and current are guaranteed to be on the identical feature definition.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from src.api.featurize import featurize

PREDICTION_COL = "prediction"


def read_recent(audit_log: Path, limit: int = 500) -> list[dict]:
    """Load the most recent prediction records from the JSONL audit log."""
    path = Path(audit_log)
    if not path.exists():
        return []
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # tolerate a torn final line
    return records[-limit:]


def current_frame(records: list[dict], spec: dict) -> pd.DataFrame:
    """Rebuild the served feature matrix (+ score) from audit-log records."""
    encounters = [r["encounter"] for r in records]
    X = featurize(encounters, spec).reset_index(drop=True)
    X[PREDICTION_COL] = [float(r["risk"]) for r in records]
    return X


def _definition(spec: dict, df: pd.DataFrame) -> DataDefinition:
    numerical = [c for c in spec["numeric_features"] if c in df.columns]
    if PREDICTION_COL in df.columns:
        numerical = numerical + [PREDICTION_COL]
    categorical = [c for c in spec["categorical_features"] if c in df.columns]
    return DataDefinition(numerical_columns=numerical, categorical_columns=categorical)


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, spec: dict):
    """Run Evidently's data-drift preset; returns the Snapshot."""
    cols = list(current.columns)
    reference = reference[cols].copy()
    current = current.copy()
    # Cast categoricals to strings on both sides so a category present in only
    # one frame can't trip the categorical drift test.
    for c in spec["categorical_features"]:
        if c in cols:
            reference[c] = reference[c].astype(str)
            current[c] = current[c].astype(str)

    definition = _definition(spec, current)
    ref_ds = Dataset.from_pandas(reference, data_definition=definition)
    cur_ds = Dataset.from_pandas(current, data_definition=definition)
    return Report(metrics=[DataDriftPreset()]).run(current_data=cur_ds, reference_data=ref_ds)
