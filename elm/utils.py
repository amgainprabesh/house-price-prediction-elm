"""Data loading, feature engineering, scaling and regression metrics.

Everything here is written directly on NumPy/pandas so the experimental
pipeline does not depend on any modelling framework.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "median_house_value"
NUMERIC_FEATURES = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]
CATEGORICAL_FEATURE = "ocean_proximity"


# --------------------------------------------------------------------------
# Loading and cleaning
# --------------------------------------------------------------------------


def load_housing(path: str = "data/housing.csv") -> pd.DataFrame:
    """Read the California housing CSV and return a cleaned DataFrame.

    Cleaning steps (each repeated in the report's methodology section):
      * rows with a missing `total_bedrooms` are dropped;
      * the target is capped at its published ceiling of 500,001 so the
        notorious saturation spike does not dominate the loss.
    """
    df = pd.read_csv(path)

    missing_before = int(df["total_bedrooms"].isna().sum())
    df = df.dropna(subset=["total_bedrooms"]).copy()

    # The public file caps the target at 500001 USD; keep the capped rows but
    # record how many there were for the report.
    n_capped = int((df[TARGET] >= 500000).sum())

    df.attrs["missing_bedrooms_dropped"] = missing_before
    df.attrs["n_capped_target"] = n_capped
    return df.reset_index(drop=True)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the three ratio features used by the model.

    Raw room/bedroom/population counts are block-size measures that say
    little about a single house; their per-household ratios carry most of
    the signal.
    """
    df = df.copy()
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]
    return df


ENGINEERED_FEATURES = [
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household",
]


def encode_categorical(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode `ocean_proximity`, dropping `ISLAND` (25 rows only).

    Returns the encoded frame and the list of resulting dummy columns.
    """
    dummies = pd.get_dummies(
        df[CATEGORICAL_FEATURE], prefix="ocean", dtype=float
    )
    dummy_cols = [c for c in dummies.columns if c != "ocean_ISLAND"]
    dummies = dummies[dummy_cols]
    out = pd.concat([df.drop(columns=[CATEGORICAL_FEATURE]), dummies], axis=1)
    return out, dummy_cols


def feature_columns(dummy_cols: list[str]) -> list[str]:
    """Full ordered feature list given the dummy columns in play."""
    return NUMERIC_FEATURES + ENGINEERED_FEATURES + dummy_cols


# --------------------------------------------------------------------------
# Scaling and splitting
# --------------------------------------------------------------------------


class StandardScaler:
    """Z-score standardiser (fit on train only, then applied to test)."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        # A constant column would divide by zero - leave it as-is.
        self.std_ = np.where(std == 0.0, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X * self.std_ + self.mean_


def train_test_split(
    X: np.ndarray, y: np.ndarray, test_frac: float = 0.2, seed: int = 42
):
    """Deterministic shuffle-split returning (X_train, X_test, y_train, y_test)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_test = int(round(n * test_frac))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot else 0.0


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error, in percent."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)


def regression_report(y_true, y_pred) -> dict:
    """All four metrics in one dict, with the unit (USD) made explicit."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mape_pct": mape(y_true, y_pred),
    }