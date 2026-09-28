"""Fair ELM-versus-back-propagation study plus a clean timing study.

Run from the project root:

    .venv/Scripts/python.exe scripts/convergence_study.py

Writes
------
models/timing_study.json       single-solve training time for each L
models/backprop_study.json     test accuracy per epoch for an identical MLP
report/figs/fig_timing.png
report/figs/fig_backprop_curve.png
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elm import ELMRegressor  # noqa: E402
from elm.utils import rmse  # noqa: E402

HIDDEN = [64, 128, 256, 512, 1024]
BUDGETS = [25, 50, 100, 200, 300]


def timing_study(Xtr, ytr, repeats=3):
    """Time the one-shot solution for several hidden-layer sizes.

    Each width is fitted ``repeats`` times and the median is reported, because a
    single wall-clock reading on a shared machine varies by more than the effect
    being measured.
    """
    rows = []
    for L in HIDDEN:
        times = []
        for _ in range(repeats):
            m = ELMRegressor(n_hidden=L, activation="sigmoid", random_state=42)
            t0 = time.perf_counter()
            m.fit(Xtr, ytr)
            times.append(time.perf_counter() - t0)
        med = float(np.median(times))
        rows.append({"n_hidden": L, "fit_seconds": med,
                     "repeats": repeats, "all_seconds": times})
        print(f"   L={L:>5}  median={med:7.2f}s  (runs: "
              f"{', '.join(f'{t:.2f}' for t in times)})")
    return rows


def backprop_study(Xtr, ytr, Xte, yte, L=512):
    """Train an identical single-hidden-layer network at several epoch budgets.

    The architecture (one hidden layer of L logistic units, linear output) is
    identical to the ELM; only the learning algorithm differs.  Each point of
    the curve is an INDEPENDENT fit from the same seed with a different
    ``max_iter`` budget, so the curve is the test error as a function of the
    training budget and its last point is directly comparable with the
    back-propagation baseline of the experiment script (same optimiser, same
    learning rate, same epochs).
    """
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler as SkScaler

    sk = SkScaler().fit(Xtr)
    Ztr, Zte = sk.transform(Xtr), sk.transform(Xte)

    # The target is z-scored for the gradient-descent run. This is standard
    # practice for back-propagation (an unscaled target of order 10^5 makes the
    # first updates meaningless) and it gives the MLP its best chance, so the
    # comparison against the ELM is a fair one.
    y_mean, y_std = float(ytr.mean()), float(ytr.std())
    Ttr = (ytr - y_mean) / y_std
    Tte = (yte - y_mean) / y_std

    curve, t0 = [], time.perf_counter()
    for epochs in BUDGETS:
        mlp = MLPRegressor(hidden_layer_sizes=(L,), activation="logistic",
                           solver="adam", learning_rate_init=1e-3,
                           max_iter=epochs, random_state=42)
        mlp.fit(Ztr, Ttr)
        elapsed = time.perf_counter() - t0
        test_rmse = rmse(yte, mlp.predict(Zte) * y_std + y_mean)
        train_rmse = rmse(ytr, mlp.predict(Ztr) * y_std + y_mean)
        curve.append({
            "epoch": epochs,
            "elapsed_seconds": elapsed,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
        })
        print(f"   epoch {epochs:>4}  {elapsed:7.1f}s  "
              f"train={train_rmse:9.0f}  test={test_rmse:9.0f}")
    return curve, BUDGETS[-1]


def main() -> None:
    d = np.load("models/splits.npz", allow_pickle=False)
    Xtr, Xte = d["X_train"], d["X_test"]
    ytr, yte = d["y_train"], d["y_test"]

    print("[A] single-solve timing")
    timing = timing_study(Xtr, ytr)
    with open("models/timing_study.json", "w", encoding="utf-8") as fh:
        json.dump(timing, fh, indent=2)

    print("\n[B] back-propagation learning curve (identical 512-unit network)")
    curve, epochs = backprop_study(Xtr, ytr, Xte, yte, L=512)
    payload = {
        "hidden_units": 512,
        "activation": "logistic (sigmoid)",
        "optimiser": "adam, learning rate 1e-3",
        "epochs_run": epochs,
        "curve": curve,
        "final_test_rmse": curve[-1]["test_rmse"],
        "final_elapsed_seconds": curve[-1]["elapsed_seconds"]
        - curve[-2]["elapsed_seconds"],
        "total_study_seconds": curve[-1]["elapsed_seconds"],
    }
    with open("models/backprop_study.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nbackprop: {epochs} epochs, "
          f"{payload['final_elapsed_seconds']:.1f}s for the final budget, "
          f"test RMSE {curve[-1]['test_rmse']:,.0f}")
    print("wrote models/timing_study.json, models/backprop_study.json")


if __name__ == "__main__":
    main()