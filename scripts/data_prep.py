"""Prepare the California housing data: clean, engineer, split, scale, save.

Run from the project root:

    .venv/Scripts/python.exe scripts/data_prep.py

Writes
------
data/train.csv        cleaned training rows (raw units, for the report)
data/test.csv         cleaned test rows
models/scaler.npz     mean/std used for z-scoring the feature matrix
models/splits.npz     the exact scaled arrays the models consume
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elm.utils import (  # noqa: E402
    TARGET,
    add_engineered_features,
    encode_categorical,
    feature_columns,
    load_housing,
    StandardScaler,
    train_test_split,
)

TEST_FRAC = 0.20
SEED = 42


def main() -> None:
    df = load_housing("data/housing.csv")
    n_dropped = int(df.attrs["missing_bedrooms_dropped"])
    n_capped = int(df.attrs["n_capped_target"])
    print(f"rows after cleaning      : {len(df)}")
    print(f"  dropped (no bedrooms)  : {n_dropped}")
    print(f"  target at 500k ceiling : {n_capped}")

    df = add_engineered_features(df)
    df, dummy_cols = encode_categorical(df)
    feats = feature_columns(dummy_cols)
    print(f"features ({len(feats)})          : {', '.join(feats)}")

    X = df[feats].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, TEST_FRAC, SEED)

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    print(f"train / test            : {len(X_tr)} / {len(X_te)}")

    os.makedirs("models", exist_ok=True)
    np.savez(
        "models/splits.npz",
        X_train=X_tr_s,
        X_test=X_te_s,
        y_train=y_tr,
        y_test=y_te,
        features=np.array(feats),
    )
    np.savez("models/scaler.npz", mean=scaler.mean_, std=scaler.std_,
             features=np.array(feats))

    # Raw-unit copies for inspection / the report's sample table.
    df.iloc[: len(X_tr)].to_csv("data/train.csv", index=False)
    df.iloc[len(X_tr):].to_csv("data/test.csv", index=False)

    summary = {
        "rows_clean": int(len(df)),
        "rows_dropped_missing_bedrooms": n_dropped,
        "rows_capped_target": n_capped,
        "n_features": len(feats),
        "features": feats,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "target_mean": float(y.mean()),
        "target_std": float(y.std()),
        "target_min": float(y.min()),
        "target_max": float(y.max()),
        "test_frac": TEST_FRAC,
        "seed": SEED,
    }
    with open("models/data_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("wrote models/splits.npz, models/scaler.npz, models/data_summary.json")


if __name__ == "__main__":
    main()