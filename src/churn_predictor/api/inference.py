"""Carga do modelo e função de predição para a API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from churn_predictor.config import API_DECISION_THRESHOLD, MODEL_PATH, MODEL_VERSION
from churn_predictor.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ModelState:
    """Estado do modelo carregado na API."""

    pipeline: Pipeline
    threshold: float
    model_version: str


_STATE: ModelState | None = None

# Colunas que sofreram transformação no pipeline
INTERNET_DEPENDENT_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def load_model_artifacts(threshold: float | None = None) -> ModelState:
    """Carrega o pipeline no disco e cacheia em memória."""

    global _STATE
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}")

    pipeline = joblib.load(MODEL_PATH)

    _STATE = ModelState(
        pipeline=pipeline,
        threshold=threshold if threshold is not None else API_DECISION_THRESHOLD,
        model_version=MODEL_VERSION,
    )

    log.info(
        "model_loaded",
        version=_STATE.model_version,
        threshold=_STATE.threshold,
        classifier=type(pipeline.named_steps["classifier"]).__name__,
    )
    return _STATE


def get_state() -> ModelState:
    """Retorna o estado do modelo carregado. Lança erro se não tiver sido carregado."""
    if _STATE is None:
        raise RuntimeError("Modelo não carregado. Chame load_model_artifacts() primeiro.")
    return _STATE


def features_to_dataframe(features: dict[str, Any]) -> pd.DataFrame:
    """Converte o dicionário de features da requisição em um DataFrame para predição."""

    df = pd.DataFrame([features])

    # Garantir que as colunas dependentes de InternetService sejam "No internet service" se InternetService for "No"
    for col in INTERNET_DEPENDENT_COLS:
        if col in df.columns:
            df[col] = df[col].replace({"No internet service": "No"})

        if "MultipleLines" in df.columns:
            df["MultipleLines"] = df["MultipleLines"].replace({"No phone service": "No"})
    return df


def predict(features: dict[str, Any], state: ModelState | None = None) -> dict[str, Any]:
    """Roda 1 inferência. Retorna dict pronto pra serializar no PredictionResponse."""
    state = state or get_state()
    df = features_to_dataframe(features)

    # predict_proba retorna [[P(0), P(1)]] para 1 amostra
    proba = float(state.pipeline.predict_proba(df)[0, 1])
    pred = int(proba >= state.threshold)

    return {
        "churn_probability": proba,
        "churn_prediction": pred,
        "threshold": state.threshold,
        "model_version": state.model_version,
    }
