"""Extreme Learning Machine (ELM) for house price regression.

The package implements the ELM as described by Huang et al. (2006) from
scratch using only NumPy linear algebra - no learning library is used for
the model itself.

Modules
-------
elm.core    ELMRegressor and the activation-function registry
elm.utils   data loading, feature engineering, scaling and metrics
"""

from elm.core import ELMRegressor, get_activation, ACTIVATIONS
from elm.utils import (
    load_housing,
    add_engineered_features,
    StandardScaler,
    train_test_split,
    rmse,
    mae,
    r2_score,
    mape,
    regression_report,
)

__all__ = [
    "ELMRegressor",
    "get_activation",
    "ACTIVATIONS",
    "load_housing",
    "add_engineered_features",
    "StandardScaler",
    "train_test_split",
    "rmse",
    "mae",
    "r2_score",
    "mape",
    "regression_report",
]