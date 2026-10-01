from fastapi import FastAPI

from app.predictor import (
    get_model_info,
    metadata,
    predict_machine_failure,
)
from app.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)


app = FastAPI(
    title="MachineGuard API",
    description=(
        "Predict industrial machine-failure risk from operating measurements."
    ),
    version="1.0.0",
)


@app.get("/")
def root():
    """Return basic information about the service."""

    return {
        "name": "MachineGuard API",
        "version": "1.0.0",
        "documentation": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
def health():
    """Confirm that the API and trained model are available."""

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": metadata["model_version"],
    }


@app.get("/model-info")
def model_info():
    """Return model features, evaluation results, and dataset information."""

    return get_model_info()


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Estimate machine-failure risk from validated measurements."""

    return predict_machine_failure(
        product_type=request.product_type,
        air_temperature=request.air_temperature,
        process_temperature=request.process_temperature,
        rotational_speed=request.rotational_speed,
        torque=request.torque,
        tool_wear=request.tool_wear,
    )