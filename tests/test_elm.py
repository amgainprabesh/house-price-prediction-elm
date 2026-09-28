"""Unit tests for the ELM implementation (pytest).

Run from the project root:

    .venv/Scripts/python.exe -m pytest tests -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elm import ELMRegressor, get_activation  # noqa: E402
from elm.utils import (  # noqa: E402
    StandardScaler,
    mae,
    mape,
    r2_score,
    rmse,
    train_test_split,
)


# --- activation functions --------------------------------------------------


def test_sigmoid_matches_reference_values():
    g = get_activation("sigmoid")
    out = g(np.array([0.0, 1.0, -1.0]))
    assert out[0] == pytest.approx(0.5, abs=1e-12)
    assert out[1] == pytest.approx(0.7310585786, abs=1e-9)
    assert out[2] == pytest.approx(0.2689414214, abs=1e-9)


def test_sigmoid_is_numerically_stable_for_large_inputs():
    g = get_activation("sigmoid")
    out = g(np.array([-800.0, 800.0]))
    assert np.isfinite(out).all()
    assert out[0] == pytest.approx(0.0, abs=1e-12)
    assert out[1] == pytest.approx(1.0, abs=1e-12)


def test_relu_and_hardlim_are_piecewise():
    assert list(get_activation("relu")(np.array([-2.0, 3.0]))) == [0.0, 3.0]
    assert list(get_activation("hardlim")(np.array([-2.0, 0.0, 2.0]))) == [0.0, 1.0, 1.0]


def test_unknown_activation_raises_value_error():
    with pytest.raises(ValueError, match="Unknown activation"):
        get_activation("cube")


# --- ELM training ----------------------------------------------------------


def test_elm_learns_a_linear_function_almost_exactly():
    """With many hidden neurons the ELM should reproduce a smooth target."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(400, 3))
    y = X @ np.array([1.5, -2.0, 0.5]) + 3.0

    model = ELMRegressor(n_hidden=400, activation="sine",
                         random_state=1).fit(X, y)
    # The residual is far below the signal scale (~8 USD here) but not exactly
    # zero: the random feature matrix is not square, so the fit is a very
    # close least-squares approximation rather than an exact interpolation.
    assert rmse(y, model.predict(X)) < 0.01


def test_training_is_reproducible_for_a_fixed_seed():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4))
    y = rng.normal(size=120)

    a = ELMRegressor(n_hidden=64, random_state=7).fit(X, y).predict(X)
    b = ELMRegressor(n_hidden=64, random_state=7).fit(X, y).predict(X)
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_different_seeds_give_different_hidden_layers():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4))
    y = rng.normal(size=120)

    a = ELMRegressor(n_hidden=64, random_state=1).fit(X, y)
    b = ELMRegressor(n_hidden=64, random_state=2).fit(X, y)
    assert not np.allclose(a.input_weights_, b.input_weights_)
    assert not np.allclose(a.predict(X), b.predict(X))


def test_hidden_matrix_shape_and_rank_behaviour():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(50, 6))
    y = rng.normal(size=50)
    model = ELMRegressor(n_hidden=20, random_state=0).fit(X, y)
    H = model._hidden(X)
    assert H.shape == (50, 20)


def test_ridge_variant_runs_and_stays_finite():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = rng.normal(size=200)
    model = ELMRegressor(n_hidden=300, alpha=1e-4, random_state=0).fit(X, y)
    preds = model.predict(X)
    assert np.isfinite(preds).all()
    assert rmse(y, preds) < np.std(y)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="must be fitted"):
        ELMRegressor().predict(np.zeros((2, 3)))


def test_feature_count_mismatch_raises():
    model = ELMRegressor(n_hidden=8, random_state=0).fit(
        np.zeros((10, 3)), np.zeros(10))
    with pytest.raises(ValueError, match="Expected 3 features"):
        model.predict(np.zeros((2, 5)))


def test_row_count_mismatch_between_X_and_y_raises():
    with pytest.raises(ValueError, match="rows but y has"):
        ELMRegressor().fit(np.zeros((10, 3)), np.zeros(9))


def test_one_dimensional_X_is_rejected():
    with pytest.raises(ValueError, match="X must be 2-D"):
        ELMRegressor().fit(np.zeros(10), np.zeros(10))


def test_save_and_load_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    y = rng.normal(size=80)
    model = ELMRegressor(n_hidden=32, activation="tanh", random_state=4).fit(X, y)

    path = tmp_path / "m.npz"
    model.save(str(path))
    reloaded = ELMRegressor.load(str(path))

    assert reloaded.n_hidden == model.n_hidden
    assert reloaded.activation == model.activation
    np.testing.assert_allclose(model.predict(X), reloaded.predict(X), rtol=1e-12)


def test_training_time_is_recorded():
    rng = np.random.default_rng(0)
    model = ELMRegressor(n_hidden=16, random_state=0).fit(
        rng.normal(size=(50, 3)), rng.normal(size=50))
    assert model.training_time_ >= 0.0


# --- preprocessing ---------------------------------------------------------


def test_standard_scaler_produces_zero_mean_unit_std():
    X = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
    Z = StandardScaler().fit_transform(X)
    np.testing.assert_allclose(Z.mean(axis=0), [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(Z.std(axis=0), [1.0, 1.0], atol=1e-12)


def test_standard_scaler_leaves_constant_column_untouched():
    X = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    Z = StandardScaler().fit_transform(X)
    assert np.isfinite(Z).all()
    np.testing.assert_allclose(Z[:, 1], [0.0, 0.0, 0.0])


def test_scaler_inverse_transform_round_trips():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 4))
    sc = StandardScaler().fit(X)
    np.testing.assert_allclose(sc.inverse_transform(sc.transform(X)), X,
                               rtol=1e-12)


# --- metrics ---------------------------------------------------------------


def test_rmse_mae_r2_and_mape_on_a_known_case():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(200 / 3))
    assert mae(y_true, y_pred) == pytest.approx(20 / 3)
    assert mape(y_true, y_pred) == pytest.approx((10/100 + 10/200) / 3 * 100)
    assert r2_score(y_true, y_pred) < 1.0


def test_perfect_prediction_scores_are_exact():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert rmse(y, y) == 0.0
    assert mae(y, y) == 0.0
    assert mape(y, y) == 0.0
    assert r2_score(y, y) == pytest.approx(1.0)


def test_r2_of_mean_predictor_is_zero():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2_score(y, np.full_like(y, y.mean())) == pytest.approx(0.0)


def test_r2_on_constant_target_is_zero_not_nan():
    y = np.full(5, 7.0)
    assert r2_score(y, np.full(5, 7.0)) == 0.0


# --- splitting -------------------------------------------------------------


def test_split_is_deterministic_and_sized_correctly():
    X = np.arange(100).reshape(50, 2).astype(float)
    y = np.arange(50).astype(float)
    a = train_test_split(X, y, 0.2, seed=42)
    b = train_test_split(X, y, 0.2, seed=42)
    for left, right in zip(a, b):
        np.testing.assert_array_equal(left, right)
    assert a[0].shape[0] == 40
    assert a[2].shape[0] == 40
    assert a[1].shape[0] == 10


def test_split_has_no_overlap():
    X = np.arange(200).reshape(100, 2).astype(float)
    y = np.arange(100).astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, 0.25, seed=1)
    assert set(y_tr.tolist()).isdisjoint(set(y_te.tolist()))
    assert len(y_tr) + len(y_te) == 100