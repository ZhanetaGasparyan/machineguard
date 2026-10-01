from app.predictor import get_model_info, predict_machine_failure


def test_low_risk_prediction():
    result = predict_machine_failure(
        product_type="L",
        air_temperature=298.2,
        process_temperature=308.7,
        rotational_speed=1408,
        torque=46.3,
        tool_wear=3,
    )

    assert 0 <= result["failure_probability"] <= 1
    assert result["prediction"] == 0
    assert result["risk_level"] == "low"
    assert result["model_name"] == "random_forest"
    assert result["model_version"] == "1.0.0"


def test_high_risk_prediction():
    result = predict_machine_failure(
        product_type="L",
        air_temperature=303.5,
        process_temperature=312.5,
        rotational_speed=1200,
        torque=70.0,
        tool_wear=230,
    )

    assert 0 <= result["failure_probability"] <= 1
    assert result["prediction"] == 1
    assert result["risk_level"] == "high"


def test_prediction_is_deterministic():
    inputs = {
        "product_type": "M",
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500,
        "torque": 40.0,
        "tool_wear": 100,
    }

    first_result = predict_machine_failure(**inputs)
    second_result = predict_machine_failure(**inputs)

    assert first_result == second_result


def test_model_info():
    info = get_model_info()

    assert info["model_name"] == "random_forest"
    assert info["model_version"] == "1.0.0"
    assert len(info["features"]) == 6
    assert "test_metrics" in info