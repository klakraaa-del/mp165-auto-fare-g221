
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_input():
    payload = {
        "distance_km": 5.0,
        "wait_min": 10.0,
        "passengers": 2,
        "hour": 14,
        "luggage_bags": 1,
        "fuel_price_rs": 96.5,
        "booking": "app",
        "zone": "Market"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "predicted_fare_rs" in response.json()


def test_predict_rejects_invalid_distance():
    payload = {
        "distance_km": -5,
        "wait_min": 10.0,
        "passengers": 2,
        "hour": 14,
        "luggage_bags": 1,
        "fuel_price_rs": 96.5,
        "booking": "app",
        "zone": "Market"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_fuel_price_above_limit():
    payload = {
        "distance_km": 5.0,
        "wait_min": 10.0,
        "passengers": 2,
        "hour": 14,
        "luggage_bags": 1,
        "fuel_price_rs": 151,
        "booking": "app",
        "zone": "Market"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
