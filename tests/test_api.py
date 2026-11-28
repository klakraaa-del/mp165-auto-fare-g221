from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "MP165 Auto-Rickshaw Fare Prediction API"
    assert data["group"] == "G221"
    assert data["status"] == "operational"
    assert data["version"] == "1.0.0"
    assert "endpoints" in data


def test_model_info_endpoint():
    response = client.get("/model-info")

    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "mp165-fare-model"
    assert data["status"] == "loaded"
    assert "numeric" in data["features"]
    assert "categorical" in data["features"]


def test_request_timing_header_middleware():
    response = client.get("/health")

    assert response.status_code == 200
    assert "x-process-time-ms" in response.headers
    process_time = float(response.headers["x-process-time-ms"])
    assert process_time >= 0.0


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
    data = response.json()
    assert "predicted_fare_rs" in data
    assert isinstance(data["predicted_fare_rs"], float)
    assert data["predicted_fare_rs"] > 0


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


def test_predict_rejects_invalid_booking_channel():
    payload = {
        "distance_km": 5.0,
        "wait_min": 10.0,
        "passengers": 2,
        "hour": 14,
        "luggage_bags": 1,
        "fuel_price_rs": 96.5,
        "booking": "metro_invalid",
        "zone": "Market"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_invalid_zone():
    payload = {
        "distance_km": 5.0,
        "wait_min": 10.0,
        "passengers": 2,
        "hour": 14,
        "luggage_bags": 1,
        "fuel_price_rs": 96.5,
        "booking": "app",
        "zone": "Unknown City Zone"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_invalid_hour():
    payload = {
        "distance_km": 5.0,
        "wait_min": 10.0,
        "passengers": 2,
        "hour": 25,
        "luggage_bags": 1,
        "fuel_price_rs": 96.5,
        "booking": "street",
        "zone": "Station Area"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_invalid_passengers():
    payload = {
        "distance_km": 5.0,
        "wait_min": 10.0,
        "passengers": 0,
        "hour": 12,
        "luggage_bags": 1,
        "fuel_price_rs": 96.5,
        "booking": "street",
        "zone": "Outskirts"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_negative_wait_min():
    payload = {
        "distance_km": 5.0,
        "wait_min": -15.0,
        "passengers": 2,
        "hour": 12,
        "luggage_bags": 1,
        "fuel_price_rs": 96.5,
        "booking": "stand",
        "zone": "Market"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_batch_valid():
    batch_payload = {
        "rides": [
            {
                "distance_km": 3.2,
                "wait_min": 5.0,
                "passengers": 1,
                "hour": 9,
                "luggage_bags": 0,
                "fuel_price_rs": 95.0,
                "booking": "street",
                "zone": "Station Area"
            },
            {
                "distance_km": 12.5,
                "wait_min": 20.0,
                "passengers": 3,
                "hour": 18,
                "luggage_bags": 2,
                "fuel_price_rs": 98.0,
                "booking": "app",
                "zone": "Market"
            },
            {
                "distance_km": 8.0,
                "wait_min": 10.0,
                "passengers": 2,
                "hour": 14,
                "luggage_bags": 1,
                "fuel_price_rs": 96.5,
                "booking": "stand",
                "zone": "Outskirts"
            }
        ]
    }

    response = client.post("/predict/batch", json=batch_payload)

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3
    assert len(data["predictions"]) == 3
    assert data["total_estimated_fare_rs"] == round(sum(data["predictions"]), 2)
    for p in data["predictions"]:
        assert isinstance(p, float)
        assert p > 0


def test_predict_batch_rejects_empty():
    batch_payload = {"rides": []}

    response = client.post("/predict/batch", json=batch_payload)

    assert response.status_code == 422
