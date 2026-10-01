import json
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "metadata.json"


def load_model():
    """Load the trained preprocessing and classification pipeline."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python scripts/train.py` first."
        )

    return joblib.load(MODEL_PATH)


def load_metadata():
    """Load information about the trained model."""

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Model metadata not found. Run `python scripts/train.py` first."
        )

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


model = load_model()
metadata = load_metadata()


def get_risk_level(probability):
    """Convert a failure probability into a user-friendly risk level."""

    if probability < 0.20:
        return "low"

    if probability < 0.50:
        return "medium"

    return "high"


def predict_machine_failure(
    product_type,
    air_temperature,
    process_temperature,
    rotational_speed,
    torque,
    tool_wear,
):
    """Predict machine-failure risk from operating measurements."""

    input_data = pd.DataFrame(
        [
            {
                "Type": product_type,
                "Air temperature [K]": air_temperature,
                "Process temperature [K]": process_temperature,
                "Rotational speed [rpm]": rotational_speed,
                "Torque [Nm]": torque,
                "Tool wear [min]": tool_wear,
            }
        ]
    )

    failure_probability = float(model.predict_proba(input_data)[0, 1])
    threshold = float(metadata["decision_threshold"])
    prediction = int(failure_probability >= threshold)

    return {
        "failure_probability": round(failure_probability, 6),
        "prediction": prediction,
        "risk_level": get_risk_level(failure_probability),
        "decision_threshold": threshold,
        "model_name": metadata["model_name"],
        "model_version": metadata["model_version"],
    }


def get_model_info():
    """Return public information about the deployed model."""

    return {
        "model_name": metadata["model_name"],
        "model_version": metadata["model_version"],
        "decision_threshold": metadata["decision_threshold"],
        "features": metadata["features"],
        "test_metrics": metadata["test_metrics"],
        "dataset": metadata["dataset"],
    }