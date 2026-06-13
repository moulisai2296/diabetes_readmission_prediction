"""Export the registered model into standalone serving artifacts.

The API must not depend on the MLflow tracking server / sqlite db at runtime
(we don't want to ship mlruns into the Docker image). This script pulls the
registered model once and writes a self-contained `artifacts/` folder:

  artifacts/model.joblib         the calibrated XGBoost pipeline
  artifacts/feature_spec.json    feature order, category vocabularies, threshold
  artifacts/drift_reference.parquet  sample of the training feature matrix + the
                                 model's score on it, used as the /drift baseline

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


def build_drift_reference(model, feature_order: list[str], n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Sample of the *training* feature matrix + the model's score, as the drift baseline.

    Restricted to the train split (never val/test) so /drift compares live traffic
    against exactly what the model learned from.
    """
    feats = pd.read_parquet(ROOT / "data_folder/processed/features.parquet")
    train_ids = pd.read_parquet(
        ROOT / "data_folder/processed/splits/train.parquet", columns=["encounter_id"]
    )["encounter_id"]
    ref = feats[feats["encounter_id"].isin(set(train_ids))]
    ref = ref.sample(min(n, len(ref)), random_state=seed)
    X = ref[feature_order]
    out = X.copy()
    out["prediction"] = model.predict_proba(X)[:, 1]
    return out


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

    reference = build_drift_reference(model, spec["feature_order"])
    reference.to_parquet(out / "drift_reference.parquet", index=False)
    log.info("wrote drift_reference.parquet (%d rows)", len(reference))


if __name__ == "__main__":
    main()
