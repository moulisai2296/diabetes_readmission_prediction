"""Patient-grouped train/val/test split.

23.5% of patients have multiple encounters (46% of all rows) — a random row split
would put the same patient in train and test, leaking patient-specific patterns.
We split by patient_nbr: every encounter of a patient lands in exactly one split.

70/15/15 by patient groups, fixed seed. The TEST split is touched exactly once,
after the model and threshold are chosen on train/val.

Run:  uv run python -m src.models.split
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

log = logging.getLogger("split")

SEED = 42


def split_by_patient(
    df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15
) -> dict[str, pd.DataFrame]:
    groups = df["patient_nbr"]

    holdout_frac = val_size + test_size
    outer = GroupShuffleSplit(n_splits=1, test_size=holdout_frac, random_state=SEED)
    train_idx, holdout_idx = next(outer.split(df, groups=groups))
    train, holdout = df.iloc[train_idx], df.iloc[holdout_idx]

    inner = GroupShuffleSplit(
        n_splits=1, test_size=test_size / holdout_frac, random_state=SEED
    )
    val_idx, test_idx = next(inner.split(holdout, groups=holdout["patient_nbr"]))
    splits = {"train": train, "val": holdout.iloc[val_idx], "test": holdout.iloc[test_idx]}

    # invariant: a patient appears in exactly one split
    seen: set = set()
    for name, part in splits.items():
        patients = set(part["patient_nbr"])
        assert not patients & seen, f"patient leakage into {name}"
        seen |= patients
        log.info(
            "%-5s %s rows  %s patients  target rate %.4f",
            name, f"{len(part):,}", f"{len(patients):,}", part["target"].mean(),
        )
    return splits


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path,
                    default=root / "data_folder/processed/features.parquet")
    ap.add_argument("--out-dir", type=Path, default=root / "data_folder/processed/splits")
    args = ap.parse_args()

    df = pd.read_parquet(args.inp)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in split_by_patient(df).items():
        part.to_parquet(args.out_dir / f"{name}.parquet", index=False)
    log.info("wrote splits to %s", args.out_dir)


if __name__ == "__main__":
    main()
