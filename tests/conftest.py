"""Fixtures compartilhadas dos testes."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from churn_predictor.config import INTERIM_DIR, MODEL_PATH

@pytest.fixture(scope="session")
def clean_dataset_path() -> Path:
    """Caminho do parquet limpo gerado pela EDA. Skip se ausente."""
    path = INTERIM_DIR / "telco_clean.parquet"
    if not path.exists():
        pytest.skip(
            f"Parquet limpo nao encontrado em {path}. "
            "Rode notebooks/01_eda.ipynb antes."
        )
    return path


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    """TestClient da API com lifespan ativado.

    Skip se final.joblib estiver ausente — testes de API exigem modelo
    carregado, modo degraded e suas particularidades sao testados a parte.
    """
    if not MODEL_PATH.exists():
        pytest.skip(
            f"Modelo nao encontrado em {MODEL_PATH}. "
            "Rode notebooks/02_modelos.ipynb antes."
        )

    # Import dentro da fixture: evita disparar carga do FastAPI na coleta
    from churn_predictor.api.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_payload() -> dict:
    """Payload realista de 1 cliente Telco no formato RAW (19 campos).

    Cliente 'average': fibra, mensal, 12 meses de contrato, eletronic check.
    Combinacao internamente consistente: tenure x MonthlyCharges = TotalCharges.
    """
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.50,
        "TotalCharges": 1026.00,
    }