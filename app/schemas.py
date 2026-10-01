from typing import Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Measurements required to estimate machine-failure risk."""

    product_type: Literal["L", "M", "H"] = Field(
        description="Product quality type: low, medium, or high."
    )
    air_temperature: float = Field(
        gt=0,
        le=500,
        description="Air temperature in Kelvin.",
    )
    process_temperature: float = Field(
        gt=0,
        le=600,
        description="Process temperature in Kelvin.",
    )
    rotational_speed: int = Field(
        gt=0,
        le=10000,
        description="Rotational speed in revolutions per minute.",
    )
    torque: float = Field(
        ge=0,
        le=500,
        description="Torque in Newton-metres.",
    )
    tool_wear: int = Field(
        ge=0,
        le=1000,
        description="Accumulated tool wear in minutes.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "product_type": "L",
                    "air_temperature": 303.5,
                    "process_temperature": 312.5,
                    "rotational_speed": 1200,
                    "torque": 70.0,
                    "tool_wear": 230,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Machine-failure prediction returned by the API."""

    failure_probability: float
    prediction: Literal[0, 1]
    risk_level: Literal["low", "medium", "high"]
    decision_threshold: float
    model_name: str
    model_version: str


class HealthResponse(BaseModel):
    """API health-check response."""

    status: str
    model_loaded: bool
    model_version: str