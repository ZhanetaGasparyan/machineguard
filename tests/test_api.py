from fastapi.testclient import TestClient

from app.api import app


client = TestClient(app)


VALID_PAYLOAD = {
    "product_type": "L",
    "air_temperature": 303.5,
    "process_temperature": 312.5,
    "rotational_speed": 1200,
    "torque": 70.0,
    "tool_wear": 230,
}


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["name"] == "MachineGuard API"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["model_version"] == "1.0.0"


def test_model_info_endpoint():
    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["model_name"] == "random_forest"


def test_valid_prediction():
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200

    body = response.json()

    assert 0 <= body["failure_probability"] <= 1
    assert body["prediction"] in [0, 1]
    assert body["risk_level"] in ["low", "medium", "high"]


def test_invalid_product_type_is_rejected():
    invalid_payload = {
        **VALID_PAYLOAD,
        "product_type": "X",
    }

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422


def test_negative_torque_is_rejected():
    invalid_payload = {
        **VALID_PAYLOAD,
        "torque": -10,
    }

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422


def test_missing_field_is_rejected():
    incomplete_payload = {
        key: value
        for key, value in VALID_PAYLOAD.items()
        if key != "tool_wear"
    }

    response = client.post("/predict", json=incomplete_payload)

    assert response.status_code == 422