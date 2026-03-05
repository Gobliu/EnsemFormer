"""Generate reproducible train/val/test split columns in the dataset CSV.

Each split column is named ``split_{i}`` and contains ``"train"``, ``"val"``,
or ``"test"`` for every row.  Split ``i`` uses ``seed + i`` as its RNG seed so
all splits are independently reproducible.

Usage:
    python scripts/generate_splits.py
    python scripts/generate_splits.py --csv data/CycPeptMPDB-4D.csv --n_splits 10 --seed 42
"""

import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def generate_splits(
    csv_path: Path,
    n_splits: int,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    overwrite: bool,
) -> None:
    df = pd.read_csv(csv_path)
    n = len(df)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    changed = False
    for i in range(n_splits):
        col = f"split_{i}"
        if col in df.columns and not overwrite:
            logging.warning(f"Column '{col}' already exists — skipping (use --overwrite to replace).")
            continue

        rng = np.random.default_rng(seed + i)
        idx = rng.permutation(n)
        labels = np.empty(n, dtype=object)
        labels[idx[:n_train]] = "train"
        labels[idx[n_train:n_train + n_val]] = "val"
        labels[idx[n_train + n_val:]] = "test"
        df[col] = labels

        counts = {v: (labels == v).sum() for v in ("train", "val", "test")}
        logging.info(f"{col}: train={counts['train']}, val={counts['val']}, test={counts['test']}")
        changed = True

    if changed:
        df.to_csv(csv_path, index=False)
        logging.info(f"Wrote {csv_path}")
    else:
        logging.info("No columns written.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=str(_REPO_ROOT / "data" / "CycPeptMPDB-4D.csv"),
                        help="Path to the dataset CSV.")
    parser.add_argument("--n_splits", type=int, default=10, help="Number of split columns to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed; split i uses seed+i.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train fraction.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Val fraction (test = remainder).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing split columns.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = _REPO_ROOT / csv_path

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    generate_splits(csv_path, args.n_splits, args.seed, args.train_ratio, args.val_ratio, args.overwrite)


if __name__ == "__main__":
    main()
