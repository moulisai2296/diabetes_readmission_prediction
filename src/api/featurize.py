"""Turn API request records into a model-ready feature frame — the serving side
of the training/serving contract.

Reuses `engineer()` from the training pipeline verbatim, then applies the *learned*
rare-category vocabulary and sets pandas `category` dtypes with the exact training
categories (so XGBoost's native categorical codes line up with what it trained on).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.features.build_features import engineer


def load_spec(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def featurize(records: list[dict], spec: dict) -> pd.DataFrame:
    """records: list of by-alias request dicts (original column names, incl. hyphens)."""
    df = engineer(pd.DataFrame(records))

    # apply the rare-category vocabulary learned at training time
    for col in spec["lumped_features"]:
        keep = set(spec["categorical_features"][col])
        df[col] = df[col].where(df[col].isin(keep), "Other")

    # exact training categories -> identical category codes at inference
    for col, cats in spec["categorical_features"].items():
        df[col] = pd.Categorical(df[col], categories=cats)
    for col in spec["numeric_features"]:
        df[col] = pd.to_numeric(df[col])

    return df[spec["feature_order"]]
