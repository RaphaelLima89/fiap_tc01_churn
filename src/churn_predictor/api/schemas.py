"""Schemas Pydantic da API"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    """Features de 1 cliente, formato RAW do Telco Customer Churn (IBM)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 100.85,
                "TotalCharges": 100.85,
            }
        }
    )

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1] = Field(..., description="0 = não senior, 1 = senior (>=65)")
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0, le=120, description="Meses de contrato")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(..., ge=0, le=500)
    TotalCharges: float = Field(..., ge=0, le=40000)


class PredictionResponse(BaseModel):
    """Resposta de POST /predict."""

    request_id: str = Field(..., description="UUID gerado pelo middleware, ecoa em X-Request-ID")
    churn_probability: float = Field(..., ge=0, le=1, description="P(churn=1)")
    churn_prediction: Literal[0, 1] = Field(
        ..., description="Predição binária usando o threshold ativo"
    )
    threshold: float = Field(..., ge=0, le=1, description="Threshold de decisão usado")
    model_version: str = Field(..., description="Identificador do modelo carregado")
    latency_ms: float = Field(..., ge=0, description="Latência interna do endpoint (ms)")


class HealthResponse(BaseModel):
    """Resposta de GET /health. Sempre HTTP 200, mesmo em degraded."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None
    threshold: float | None = None
