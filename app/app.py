"""House Price Prediction with an Extreme Learning Machine - web application.

Run from the project root:

    .venv/Scripts/python.exe app/app.py

Then open http://127.0.0.1:5000/ in a browser.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from flask import Flask, jsonify, render_template, request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from elm import ELMRegressor  # noqa: E402

APP = Flask(__name__)
MODEL = None
SCALER = None
META = {}


def load_artefacts() -> None:
    """Load the trained ELM and the scaler fitted on the training split."""
    global MODEL, SCALER, META
    try:
        MODEL = ELMRegressor.load(os.path.join(PROJECT_ROOT, "models/elm_final.npz"))
        with np.load(os.path.join(PROJECT_ROOT, "models/scaler.npz"),
                     allow_pickle=False) as d:
            SCALER = {"mean": d["mean"], "std": d["std"],
                      "features": [str(f) for f in d["features"]]}
        with open(os.path.join(PROJECT_ROOT, "models/model_meta.json"),
                  encoding="utf-8") as fh:
            META = json.load(fh)
    except (OSError, KeyError) as exc:  # pragma: no cover - misconfiguration
        raise RuntimeError(
            "The model artefact is missing or unreadable. Train the model "
            "first with scripts/run_experiments.py. Underlying error: " + str(exc)
        ) from exc


# Load at import time, so the application is ready however it is started: a
# direct run, `flask --app app.app run`, or any WSGI server. Loading only under
# __main__ would leave the routes serving errors when launched any other way.
load_artefacts()


OCEANS = ["<1H OCEAN", "INLAND", "NEAR BAY", "NEAR OCEAN"]


def build_feature_vector(form: dict) -> np.ndarray:
    """Turn the submitted form values into the model's 15-column row.

    The three ratio features are recomputed here exactly as in training so
    the app never asks the user to type a derived quantity.
    """
    rooms = float(form["total_rooms"])
    bedrooms = float(form["total_bedrooms"])
    population = float(form["population"])
    households = float(form["households"])
    if households <= 0:
        raise ValueError("Households must be greater than zero.")

    row = [
        float(form["longitude"]),
        float(form["latitude"]),
        float(form["housing_median_age"]),
        rooms,
        bedrooms,
        population,
        households,
        float(form["median_income"]),
        rooms / households,
        bedrooms / rooms if rooms else 0.0,
        population / households,
    ]
    ocean = form.get("ocean_proximity", "INLAND")
    row += [1.0 if ocean == o else 0.0 for o in OCEANS]
    return np.array(row, dtype=float).reshape(1, -1)


def predict_price(features: np.ndarray) -> float:
    """Scale the row with the training statistics and predict in USD."""
    scaled = (features - SCALER["mean"]) / SCALER["std"]
    return float(MODEL.predict(scaled)[0])


@APP.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        oceans=OCEANS,
        meta=META,
        prediction=None,
        form={},
        error=None,
    )


@APP.route("/predict", methods=["POST"])
def predict():
    """Form submission route: validate, predict, re-render the page."""
    form = request.form.to_dict()
    try:
        features = build_feature_vector(form)
        price = predict_price(features)
    except (KeyError, ValueError) as exc:
        return render_template(
            "index.html", oceans=OCEANS, meta=META,
            prediction=None, form=form, error=str(exc),
        ), 400

    return render_template(
        "index.html",
        oceans=OCEANS,
        meta=META,
        prediction={
            "price": price,
            "low": price * 0.85,
            "high": price * 1.15,
        },
        form=form,
        error=None,
    )


@APP.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON endpoint used by the automated system tests."""
    payload = request.get_json(silent=True) or {}
    try:
        features = build_feature_vector(payload)
        price = predict_price(features)
    except (KeyError, ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({
        "predicted_price_usd": round(price, 2),
        "model": {
            "algorithm": "Extreme Learning Machine",
            "hidden_neurons": META.get("n_hidden"),
            "activation": META.get("activation"),
        },
    })


@APP.route("/model-info", methods=["GET"])
def model_info():
    """Model card page: shows the recorded test-set performance."""
    results_path = os.path.join(PROJECT_ROOT, "models/results.json")
    results = {}
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as fh:
            r = json.load(fh)
        results = {
            "baselines": r.get("baselines", {}),
            "final_model": r.get("final_model", {}),
            "seed_stability": r.get("seed_stability", {}),
        }
    return jsonify({"meta": META, "results": results})


if __name__ == "__main__":
    load_artefacts()
    print(f"ELM ready: L={META['n_hidden']} "
          f"activation={META['activation']} "
          f"test R2={META['test_r2']:.4f}")
    APP.run(host="127.0.0.1", port=5000, debug=False)