"""Export the registered model into standalone serving artifacts.

The API must not depend on the MLflow tracking server / sqlite db at runtime
(we don't want to ship mlruns into the Docker image). This script pulls the
registered model once and writes a self-contained `artifacts/` folder:

  artifacts/model.joblib       the calibrated XGBoost pipeline
  artifacts/feature_spec.json  feature order, category vocabularies, threshold

The category vocabularies are the exact training categories — serving must set
identical pandas `category` dtypes or XGBoost's native categorical codes shift
and predictions go silently wrong.

Run (after train + finalize have registered the model):
  uv run python -m src.api.export_model
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import mlflow
import pandas as pd

from src.features.build_features import CATEGORICAL_COLS, RARE_LUMP_COLS

log = logging.getLogger("export")

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "readmission-risk"
NON_FEATURE = ["encounter_id", "patient_nbr", "target"]


def build_feature_spec() -> dict:
    feats = pd.read_parquet(ROOT / "data_folder/processed/features.parquet")
    feature_order = [c for c in feats.columns if c not in NON_FEATURE]
    categorical = {
        c: [str(v) for v in feats[c].cat.categories]
        for c in CATEGORICAL_COLS
    }
    numeric = [c for c in feature_order if c not in categorical]
    report = json.loads((ROOT / "src/models/final_model_report.json").read_text())
    return {
        "feature_order": feature_order,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "lumped_features": RARE_LUMP_COLS,
        "threshold": report["threshold"],
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")

    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name = '{MODEL_NAME}'")
    latest = max(versions, key=lambda v: int(v.version))
    log.info("loading %s v%s", MODEL_NAME, latest.version)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{latest.version}")

    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    joblib.dump(model, out / "model.joblib")

    spec = build_feature_spec()
    spec["model_version"] = latest.version
    (out / "feature_spec.json").write_text(json.dumps(spec, indent=2))
    log.info("wrote %s (%d features, threshold %.3f)",
             out, len(spec["feature_order"]), spec["threshold"])


if __name__ == "__main__":
    main()
