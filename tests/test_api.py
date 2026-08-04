"""API contract tests.

These verify the serving path independently of any particular model, which is
the point of the Predictor Protocol.
"""

import pytest
from fastapi.testclient import TestClient

from sdp.api.main import app, get_predictor
from sdp.model.dummy import DummyPredictor
from sdp.taxonomy import NUM_CLASSES


@pytest.fixture
def client():
    """Test client with a seeded predictor injected."""
    app.dependency_overrides[get_predictor] = lambda: DummyPredictor(seed=0)
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["num_classes"] == NUM_CLASSES


def test_predict_returns_valid_distribution(client):
    response = client.post("/predict", json={"code": "int main() { return 0; }"})
    assert response.status_code == 200
    body = response.json()

    assert len(body["probabilities"]) == NUM_CLASSES
    assert sum(p["probability"] for p in body["probabilities"]) == pytest.approx(1.0)
    assert body["predicted_class"] in [p["label"] for p in body["probabilities"]]
    assert body["is_placeholder"] is True


def test_empty_code_rejected(client):
    assert client.post("/predict", json={"code": ""}).status_code == 422


def test_missing_field_rejected(client):
    assert client.post("/predict", json={}).status_code == 422
