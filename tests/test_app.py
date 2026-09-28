"""System tests for the Flask prediction application.

These drive the app through Flask's test client - the same WSGI path a real
browser hits - so they exercise routing, validation and prediction together.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app.app as web  # noqa: E402

VALID = {
    "longitude": "-118.25",
    "latitude": "34.05",
    "housing_median_age": "25",
    "total_rooms": "2400",
    "total_bedrooms": "480",
    "population": "1200",
    "households": "450",
    "median_income": "5.5",
    "ocean_proximity": "NEAR OCEAN",
}


@pytest.fixture()
def client():
    web.load_artefacts()
    web.APP.config.update(TESTING=True)
    with web.APP.test_client() as c:
        yield c


def test_module_import_alone_loads_the_artefacts():
    """A WSGI server imports the module; it never runs __main__.

    So the artefacts must be loaded at import time, otherwise every route
    serves an error under `flask --app app.app run` while the test client,
    which calls load_artefacts() itself, still passes.
    """
    import importlib

    reloaded = importlib.reload(web)
    assert reloaded.META.get("n_hidden"), "metadata not loaded at import time"
    assert reloaded.MODEL is not None, "model not loaded at import time"
    assert reloaded.SCALER is not None, "scaler not loaded at import time"


def test_home_page_renders_with_model_card(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "Extreme Learning Machine" in body
    assert str(web.META["n_hidden"]) in body


def test_predict_returns_a_price_for_valid_input(client):
    r = client.post("/predict", data=VALID)
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "Estimated median house value" in body
    assert "$" in body


def test_prediction_is_in_a_plausible_range(client):
    r = client.post("/api/predict", json=VALID)
    assert r.status_code == 200
    price = r.get_json()["predicted_price_usd"]
    # The training target spans 15k - 500k USD; any prediction must fall in a
    # wide but still sane envelope around that.
    assert 10_000 < price < 600_000


def test_increasing_income_does_not_decrease_the_prediction(client):
    """Monotonic sanity check on the dominant feature."""
    low = dict(VALID, median_income="2.0")
    high = dict(VALID, median_income="9.0")
    p_low = client.post("/api/predict", json=low).get_json()["predicted_price_usd"]
    p_high = client.post("/api/predict", json=high).get_json()["predicted_price_usd"]
    assert p_high > p_low


def test_expensive_coastal_row_predicts_above_inland_row(client):
    coast = dict(VALID, ocean_proximity="NEAR BAY", latitude="37.8",
                 longitude="-122.4")
    inland = dict(VALID, ocean_proximity="INLAND", latitude="36.5",
                  longitude="-119.5")
    p_coast = client.post("/api/predict", json=coast).get_json()["predicted_price_usd"]
    p_inland = client.post("/api/predict", json=inland).get_json()["predicted_price_usd"]
    assert p_coast > p_inland


def test_missing_field_is_rejected_with_400(client):
    payload = {k: v for k, v in VALID.items() if k != "median_income"}
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_zero_households_is_rejected(client):
    r = client.post("/api/predict", json=dict(VALID, households="0"))
    assert r.status_code == 400
    assert "Households" in r.get_json()["error"]


def test_non_numeric_input_is_rejected(client):
    r = client.post("/api/predict", json=dict(VALID, total_rooms="many"))
    assert r.status_code == 400


def test_form_route_reports_validation_error_in_page(client):
    r = client.post("/predict", data=dict(VALID, households="0"))
    assert r.status_code == 400
    assert "Households" in r.get_data(as_text=True)


def test_model_info_endpoint_exposes_recorded_metrics(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    payload = r.get_json()
    assert payload["meta"]["n_hidden"] > 0
    assert 0.0 < payload["meta"]["test_r2"] <= 1.0
    assert "baselines" in payload["results"]


def test_same_input_gives_identical_prediction(client):
    a = client.post("/api/predict", json=VALID).get_json()["predicted_price_usd"]
    b = client.post("/api/predict", json=VALID).get_json()["predicted_price_usd"]
    assert a == b


def test_ratio_features_are_computed_not_typed(client):
    """The app must derive rooms_per_household itself."""
    row = web.build_feature_vector(VALID)
    assert row.shape == (1, len(web.META["features"]))
    idx = web.META["features"].index("rooms_per_household")
    assert row[0, idx] == pytest.approx(2400 / 450)