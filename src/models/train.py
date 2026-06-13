"""Train and compare models for 30-day readmission risk.

Models (per the agreed plan):
- logreg : logistic regression + class weighting — the baseline to beat
- xgb    : XGBoost, tuned with patient-grouped CV
- lgbm   : LightGBM, tuned with patient-grouped CV

All runs are logged to MLflow (local ./mlruns file store). Models are fit on the
TRAIN split and compared on the VAL split; the TEST split is not touched here —
it is reserved for a single final evaluation after model + threshold are chosen.

Imbalance (11.4% positive) is handled with class weights / scale_pos_weight rather
than resampling: weighting changes the loss, not the data, keeps probabilities
re-calibratable, and avoids duplicating patients (which would interact badly with
grouped CV).

Run:  uv run python -m src.models.train            # all three
      uv run python -m src.models.train --models logreg
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

log = logging.getLogger("train")

SEED = 42
N_ITER = 20          # random-search candidates per boosted model
CV_FOLDS = 3         # patient-grouped CV folds
FIXED_PRECISION = 0.30  # report recall at this precision (~2.6x lift over 11.4% base)

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data_folder/processed/splits"
DROP_COLS = ["encounter_id", "patient_nbr", "target"]


def load_split(name: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_parquet(SPLIT_DIR / f"{name}.parquet")
    return df.drop(columns=DROP_COLS), df["target"], df["patient_nbr"]


def recall_at_precision(y_true, y_score, min_precision: float) -> float:
    """Highest recall achievable while keeping precision >= min_precision."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ok = precision >= min_precision
    return float(recall[ok].max()) if ok.any() else 0.0


def evaluate(y_true, y_score) -> dict[str, float]:
    return {
        "val_pr_auc": float(average_precision_score(y_true, y_score)),
        "val_roc_auc": float(roc_auc_score(y_true, y_score)),
        f"val_recall_at_p{int(FIXED_PRECISION * 100)}": recall_at_precision(
            y_true, y_score, FIXED_PRECISION
        ),
        "val_brier": float(brier_score_loss(y_true, y_score)),
    }


def make_logreg() -> Pipeline:
    """One-hot + scaled numerics + class-weighted logistic regression."""
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), make_column_selector(dtype_include=np.number)),
            ("cat", OneHotEncoder(handle_unknown="ignore"),
             make_column_selector(dtype_include="category")),
        ]
    )
    return Pipeline(
        [
            ("pre", pre),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=3000,
                                       random_state=SEED)),
        ]
    )


def tune(model, param_dist, X, y, groups) -> RandomizedSearchCV:
    """Random search with patient-grouped, stratified CV, optimizing PR-AUC."""
    cv = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    search = RandomizedSearchCV(
        model, param_dist, n_iter=N_ITER, scoring="average_precision",
        cv=cv, random_state=SEED, refit=True, verbose=1,
    )
    search.fit(X, y, groups=groups)
    return search


def train_logreg(X, y, groups):
    model = make_logreg()
    model.fit(X, y)
    return model, {"model_type": "logreg", "class_weight": "balanced"}


def train_xgb(X, y, groups):
    spw = (y == 0).sum() / (y == 1).sum()
    base = XGBClassifier(
        enable_categorical=True, tree_method="hist", scale_pos_weight=spw,
        eval_metric="aucpr", random_state=SEED, n_jobs=-1,
    )
    params = {
        "n_estimators": randint(200, 800),
        "learning_rate": loguniform(0.01, 0.2),
        "max_depth": randint(3, 9),
        "min_child_weight": randint(1, 11),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "reg_lambda": loguniform(0.1, 10),
    }
    search = tune(base, params, X, y, groups)
    log.info("xgb best CV PR-AUC: %.4f", search.best_score_)
    return search.best_estimator_, {"model_type": "xgb", "cv_pr_auc": search.best_score_,
                                    **search.best_params_}


def train_lgbm(X, y, groups):
    spw = (y == 0).sum() / (y == 1).sum()
    base = LGBMClassifier(
        scale_pos_weight=spw, random_state=SEED, n_jobs=-1, verbosity=-1,
    )
    params = {
        "n_estimators": randint(200, 800),
        "learning_rate": loguniform(0.01, 0.2),
        "num_leaves": randint(15, 128),
        "min_child_samples": randint(10, 100),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "reg_lambda": loguniform(0.1, 10),
    }
    search = tune(base, params, X, y, groups)
    log.info("lgbm best CV PR-AUC: %.4f", search.best_score_)
    return search.best_estimator_, {"model_type": "lgbm", "cv_pr_auc": search.best_score_,
                                    **search.best_params_}


TRAINERS = {"logreg": train_logreg, "xgb": train_xgb, "lgbm": train_lgbm}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", choices=list(TRAINERS), default=list(TRAINERS))
    args = ap.parse_args()

    # MLflow >=3.13 requires a database backend; SQLite file is local + gitignored
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("diabetes-readmission")

    X_train, y_train, groups = load_split("train")
    X_val, y_val, _ = load_split("val")
    log.info("train %s rows, val %s rows, prevalence %.4f",
             f"{len(X_train):,}", f"{len(X_val):,}", y_train.mean())

    results = {}
    for name in args.models:
        log.info("=== training %s ===", name)
        with mlflow.start_run(run_name=name):
            model, params = TRAINERS[name](X_train, y_train, groups)
            metrics = evaluate(y_val, model.predict_proba(X_val)[:, 1])

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_param("val_prevalence", round(float(y_val.mean()), 4))
            if name == "logreg":
                mlflow.sklearn.log_model(model, name="model")
            elif name == "xgb":
                mlflow.xgboost.log_model(model, name="model")
            else:
                mlflow.lightgbm.log_model(model, name="model")

            results[name] = metrics
            log.info("%s: %s", name, json.dumps(metrics, indent=2))

    print("\n=== VALIDATION COMPARISON ===")
    print(pd.DataFrame(results).T.round(4).to_string())


if __name__ == "__main__":
    main()
