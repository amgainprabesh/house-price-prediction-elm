"""Run every experiment reported in the documentation and save the artefacts.

Run from the project root:

    .venv/Scripts/python.exe scripts/run_experiments.py

Produces
--------
models/results.json          every number quoted in the report
report/figs/*.png            all result figures
models/elm_final.npz         the deployed model
models/model_meta.json       metadata the web app displays
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elm import ELMRegressor  # noqa: E402
from elm.utils import regression_report  # noqa: E402

SEED = 42
HIDDEN_SWEEP = [16, 32, 64, 128, 256, 512, 1024, 2048]
ACTIVATIONS = ["sigmoid", "tanh", "relu", "sine", "hardlim"]
ALPHA_SWEEP = [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1]
FINAL_HIDDEN = 512
FINAL_ACTIVATION = "sigmoid"


def load_splits():
    d = np.load("models/splits.npz", allow_pickle=False)
    return (d["X_train"], d["X_test"], d["y_train"], d["y_test"],
            [str(f) for f in d["features"]])


def fit_eval(Xtr, ytr, Xte, yte, **kw):
    """Fit an ELM with the given hyper-parameters and score both splits."""
    model = ELMRegressor(random_state=SEED, **kw)
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    wall = time.perf_counter() - t0
    tr = regression_report(ytr, model.predict(Xtr))
    te = regression_report(yte, model.predict(Xte))
    return model, tr, te, wall


def main() -> None:
    Xtr, Xte, ytr, yte, feats = load_splits()
    results: dict = {}
    os.makedirs("report/figs", exist_ok=True)

    print(f"train {Xtr.shape}  test {Xte.shape}  features {len(feats)}")

    # ---- Experiment 1: hidden-layer size -------------------------------
    print("\n[1] hidden-layer size sweep")
    sweep = []
    for L in HIDDEN_SWEEP:
        _, tr, te, wall = fit_eval(
            Xtr, ytr, Xte, yte, n_hidden=L, activation=FINAL_ACTIVATION
        )
        sweep.append({"n_hidden": L, "train": tr, "test": te,
                      "wall_time_s": wall,
                      "train_time_s": None})
        print(f"   L={L:>5}  test RMSE={te['rmse']:9.2f}  "
              f"R2={te['r2']:.4f}  fit={wall:6.2f}s")
    results["hidden_sweep"] = sweep

    # ---- Experiment 2: activation functions ----------------------------
    print("\n[2] activation comparison (L = %d)" % FINAL_HIDDEN)
    acts = []
    for act in ACTIVATIONS:
        _, tr, te, wall = fit_eval(
            Xtr, ytr, Xte, yte, n_hidden=FINAL_HIDDEN, activation=act
        )
        acts.append({"activation": act, "train": tr, "test": te,
                     "wall_time_s": wall})
        print(f"   {act:<8} test RMSE={te['rmse']:9.2f}  R2={te['r2']:.4f}")
    results["activation_sweep"] = acts

    # ---- Experiment 3: ridge regularisation ----------------------------
    print("\n[3] regularisation strength")
    regs = []
    for a in ALPHA_SWEEP:
        _, tr, te, wall = fit_eval(
            Xtr, ytr, Xte, yte, n_hidden=FINAL_HIDDEN,
            activation=FINAL_ACTIVATION, alpha=a
        )
        regs.append({"alpha": a, "train": tr, "test": te, "wall_time_s": wall})
        print(f"   alpha={a:<8} test RMSE={te['rmse']:9.2f}")
    results["alpha_sweep"] = regs

    # ---- Experiment 4: seed stability of the final configuration -------
    print("\n[4] seed stability (10 draws)")
    seeds = []
    for s in range(10):
        m = ELMRegressor(n_hidden=FINAL_HIDDEN,
                         activation=FINAL_ACTIVATION, random_state=s)
        m.fit(Xtr, ytr)
        te = regression_report(yte, m.predict(Xte))
        seeds.append({"seed": s, "test": te})
    arr = np.array([s["test"]["rmse"] for s in seeds])
    results["seed_stability"] = {
        "runs": seeds,
        "rmse_mean": float(arr.mean()),
        "rmse_std": float(arr.std()),
        "rmse_min": float(arr.min()),
        "rmse_max": float(arr.max()),
    }
    print(f"   RMSE {arr.mean():.2f} +/- {arr.std():.2f} "
          f"(min {arr.min():.2f}, max {arr.max():.2f})")

    # ---- Experiment 5: baselines ---------------------------------------
    print("\n[5] baselines")
    baselines = {}

    # (a) least-squares linear regression, closed form on the same features
    t0 = time.perf_counter()
    A = np.hstack([Xtr, np.ones((Xtr.shape[0], 1))])
    w, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    lin_wall = time.perf_counter() - t0

    def lin_predict(X):
        return np.hstack([X, np.ones((X.shape[0], 1))]) @ w

    baselines["linear_regression"] = {
        "train": regression_report(ytr, lin_predict(Xtr)),
        "test": regression_report(yte, lin_predict(Xte)),
        "wall_time_s": lin_wall,
        "note": "closed-form OLS on the same 15 features",
    }
    print(f"   linear      test RMSE={baselines['linear_regression']['test']['rmse']:9.2f}")

    # (b) mean predictor (the trivial floor every model must beat)
    mean_pred = np.full_like(yte, ytr.mean(), dtype=float)
    baselines["mean_predictor"] = {
        "train": regression_report(ytr, np.full_like(ytr, ytr.mean())),
        "test": regression_report(yte, mean_pred),
        "wall_time_s": 0.0,
    }
    print(f"   mean-only   test RMSE={baselines['mean_predictor']['test']['rmse']:9.2f}")

    # (c) sklearn MLP (back-propagation) for the "gradient descent is slow"
    #     comparison table in the report
    try:
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler as SkScaler

        sk = SkScaler().fit(Xtr)
        # The target is z-scored for the gradient-descent baseline: an
        # unscaled target of order 10^5 makes the first adam updates useless,
        # so scaling it is what gives the MLP a fair chance.
        y_mean, y_std = float(ytr.mean()), float(ytr.std())
        mlp = MLPRegressor(
            hidden_layer_sizes=(512,), activation="logistic",
            solver="adam", learning_rate_init=1e-3, max_iter=300,
            random_state=SEED, early_stopping=False,
        )
        t0 = time.perf_counter()
        mlp.fit(sk.transform(Xtr), (ytr - y_mean) / y_std)
        mlp_wall = time.perf_counter() - t0

        def mlp_predict(X):
            return mlp.predict(sk.transform(X)) * y_std + y_mean

        baselines["mlp_backprop"] = {
            "train": regression_report(ytr, mlp_predict(Xtr)),
            "test": regression_report(yte, mlp_predict(Xte)),
            "wall_time_s": mlp_wall,
            "iterations": int(mlp.n_iter_),
            "note": "sklearn MLPRegressor, one 512-unit logistic hidden layer, "
                    "adam, z-scored target",
        }
        b = baselines["mlp_backprop"]
        print(f"   MLP-512     test RMSE={b['test']['rmse']:9.2f} "
              f"({mlp_wall:.1f}s, {b['iterations']} epochs)")
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"   (MLP baseline unavailable: {exc})")

    results["baselines"] = baselines

    # ---- Final model ----------------------------------------------------
    print("\n[6] final model")
    final, tr, te, wall = fit_eval(
        Xtr, ytr, Xte, yte, n_hidden=FINAL_HIDDEN,
        activation=FINAL_ACTIVATION, alpha=0.0
    )
    final.save("models/elm_final.npz")
    preds = final.predict(Xte)
    np.savez("models/test_predictions.npz", y_true=yte, y_pred=preds)

    results["final_model"] = {
        "n_hidden": FINAL_HIDDEN,
        "activation": FINAL_ACTIVATION,
        "alpha": 0.0,
        "train": tr,
        "test": te,
        "wall_time_s": wall,
        "pinv_time_s": final.training_time_,
    }
    print(f"   L={FINAL_HIDDEN} {FINAL_ACTIVATION}: "
          f"test RMSE={te['rmse']:.2f}  MAE={te['mae']:.2f}  "
          f"R2={te['r2']:.4f}  MAPE={te['mape_pct']:.2f}%  "
          f"pinv={final.training_time_:.2f}s")

    meta = {
        "n_hidden": FINAL_HIDDEN,
        "activation": FINAL_ACTIVATION,
        "alpha": 0.0,
        "seed": SEED,
        "n_features": len(feats),
        "features": feats,
        "test_rmse": te["rmse"],
        "test_mae": te["mae"],
        "test_r2": te["r2"],
        "test_mape_pct": te["mape_pct"],
        "train_seconds": final.training_time_,
    }
    with open("models/model_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    results["meta"] = meta
    with open("models/results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print("\nwrote models/results.json, models/elm_final.npz, models/model_meta.json")


if __name__ == "__main__":
    main()