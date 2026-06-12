"""Finalize the production candidate: calibrate, choose threshold, register, test.

Steps (decisions agreed in Stage 3 discussion):
1. Rebuild the best XGBoost from its MLflow-logged hyperparameters and wrap it in
   isotonic calibration (CalibratedClassifierCV, patient-grouped CV folds).
   Why: class weighting fixes ranking under 11% prevalence but inflates predicted
   probabilities; the API must report honest "% risk" to the care team.
2. Derive the operating threshold from the cost model:
       flag when  p * PREVENTABLE_FRACTION * READMISSION_COST > INTERVENTION_COST
   i.e. threshold = INTERVENTION_COST / (PREVENTABLE_FRACTION * READMISSION_COST).
3. Log + register the calibrated model in the MLflow registry ("readmission-risk").
4. ONE-TIME evaluation on the held-out TEST split — model and threshold are locked
   before this script reads it.

Run:  uv run python -m src.models.finalize
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

from src.models.train import SEED, evaluate, load_split, ROOT

log = logging.getLogger("finalize")

# --- cost model (literature-based; sensitivity-checked below) -------------------
READMISSION_COST = 15_000  # avg cost of an unplanned 30-day readmission (USD)
INTERVENTION_COST = 300    # follow-up program per flagged patient (calls + visit)
PREVENTABLE_FRACTION = 0.20  # share of readmissions the intervention prevents

XGB_INT_PARAMS = {"n_estimators", "max_depth", "min_child_weight"}
XGB_FLOAT_PARAMS = {"learning_rate", "subsample", "colsample_bytree", "reg_lambda"}


def best_xgb_params() -> dict:
    """Fetch the best tuned XGBoost hyperparameters from MLflow."""
    runs = mlflow.search_runs(
        experiment_names=["diabetes-readmission"],
        filter_string="params.model_type = 'xgb'",
        order_by=["metrics.val_pr_auc DESC"],
        max_results=1,
    )
    if runs.empty:
        raise SystemExit("no xgb run found — run src.models.train first")
    raw = {c.removeprefix("params."): runs.iloc[0][c] for c in runs.columns
           if c.startswith("params.")}
    params: dict = {}
    for k, v in raw.items():
        if k in XGB_INT_PARAMS:
            params[k] = int(v)
        elif k in XGB_FLOAT_PARAMS:
            params[k] = float(v)
    log.info("best xgb params (val PR-AUC %.4f): %s",
             runs.iloc[0]["metrics.val_pr_auc"], params)
    return params


def cost_threshold(readmission_cost=READMISSION_COST, intervention_cost=INTERVENTION_COST,
                   preventable=PREVENTABLE_FRACTION) -> float:
    """Flagging pays off when p * preventable * readmission_cost > intervention_cost."""
    return intervention_cost / (preventable * readmission_cost)


def at_threshold(y_true, y_prob, thr: float) -> dict[str, float]:
    y_hat = y_prob >= thr
    flagged = float(y_hat.mean())
    return {
        "threshold": thr,
        "flag_rate": flagged,
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat)),
        # expected net savings per 1000 discharges under the cost model
        "net_savings_per_1000": float(
            1000 * (
                (y_true & y_hat).mean() * PREVENTABLE_FRACTION * READMISSION_COST
                - flagged * INTERVENTION_COST
            )
        ),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("diabetes-readmission")

    X_train, y_train, groups = load_split("train")
    X_val, y_val, _ = load_split("val")

    spw = (y_train == 0).sum() / (y_train == 1).sum()
    base = XGBClassifier(
        enable_categorical=True, tree_method="hist", scale_pos_weight=spw,
        eval_metric="aucpr", random_state=SEED, n_jobs=-1, **best_xgb_params(),
    )
    # grouped folds passed explicitly — CalibratedClassifierCV has no groups arg
    folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
                 .split(X_train, y_train, groups))
    model = CalibratedClassifierCV(base, method="isotonic", cv=folds)
    log.info("fitting calibrated model (3 grouped folds + full refit)...")
    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    val_metrics = evaluate(y_val, val_prob)
    thr = cost_threshold()
    val_at_thr = at_threshold(y_val.astype(bool), val_prob, thr)
    log.info("VAL calibrated: %s", json.dumps(val_metrics))
    log.info("VAL at threshold %.3f: %s", thr, json.dumps(val_at_thr))

    # threshold sensitivity to the cost assumptions
    sensitivity = [
        {"preventable": pv, "readmission_cost": rc,
         "threshold": round(cost_threshold(rc, INTERVENTION_COST, pv), 3)}
        for pv in (0.10, 0.20, 0.30) for rc in (10_000, 15_000, 20_000)
    ]

    with mlflow.start_run(run_name="xgb-calibrated-final"):
        mlflow.log_params({"model_type": "xgb_calibrated", "calibration": "isotonic",
                           "threshold": thr, "readmission_cost": READMISSION_COST,
                           "intervention_cost": INTERVENTION_COST,
                           "preventable_fraction": PREVENTABLE_FRACTION})
        mlflow.log_metrics(val_metrics)
        mlflow.log_metrics({f"val_thr_{k}": v for k, v in val_at_thr.items()})
        mlflow.log_dict({"sensitivity": sensitivity}, "threshold_sensitivity.json")

        fig, ax = plt.subplots(figsize=(6, 5))
        CalibrationDisplay.from_predictions(y_val, val_prob, n_bins=10, ax=ax,
                                            name="xgb isotonic-calibrated")
        ax.set_title("Calibration on validation")
        mlflow.log_figure(fig, "calibration_curve.png")

        # ---- ONE-TIME test evaluation: model + threshold locked above ----
        X_test, y_test, _ = load_split("test")
        test_prob = model.predict_proba(X_test)[:, 1]
        test_metrics = {k.replace("val_", "test_"): v
                        for k, v in evaluate(y_test, test_prob).items()}
        test_at_thr = at_threshold(y_test.astype(bool), test_prob, thr)
        mlflow.log_metrics(test_metrics)
        mlflow.log_metrics({f"test_thr_{k}": v for k, v in test_at_thr.items()})
        log.info("TEST: %s", json.dumps(test_metrics))
        log.info("TEST at threshold %.3f: %s", thr, json.dumps(test_at_thr))

        mlflow.sklearn.log_model(model, name="model",
                                 registered_model_name="readmission-risk",
                                 input_example=X_val.head(2))

    out = {
        "model": "xgboost + isotonic calibration",
        "threshold": thr,
        "cost_model": {"readmission_cost": READMISSION_COST,
                       "intervention_cost": INTERVENTION_COST,
                       "preventable_fraction": PREVENTABLE_FRACTION},
        "val": {**val_metrics, "at_threshold": val_at_thr},
        "test": {**test_metrics, "at_threshold": test_at_thr},
        "sensitivity": sensitivity,
    }
    report = ROOT / "src/models/final_model_report.json"
    report.write_text(json.dumps(out, indent=2))
    log.info("wrote %s", report)


if __name__ == "__main__":
    main()
