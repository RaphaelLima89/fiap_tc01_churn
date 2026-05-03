"""Carga da base limpa e split train/test."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from churn_predictor.config import (
    ID_COLUMN,
    INTERIM_DIR,
    RANDOM_SEED,
    TARGET_COLUMN,
    TEST_SIZE,
)
from churn_predictor.utils.logging import get_logger

logger = get_logger(__name__)

CLEAN_PARQUET = INTERIM_DIR / "telco_clean.parquet"

# Listas reaproveitadas em features/preprocessing.py
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

BINARY_CAT_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "PaperlessBilling",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

MULTI_CAT_FEATURES = ["InternetService", "Contract", "PaymentMethod"]


def load_clean_data() -> pd.DataFrame:
    """Lê data/interim/telco_clean.parquet. Raise se não existir."""
    if not CLEAN_PARQUET.exists():
        raise FileNotFoundError(
            f"Base limpa não encontrada em {CLEAN_PARQUET}. Rode o notebook 01_eda.ipynb até a §10."
        )
    df = pd.read_parquet(CLEAN_PARQUET)
    logger.info("clean_data_loaded", shape=df.shape)
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa X (features) e y (binário 0/1)."""
    y = (df[TARGET_COLUMN] == "Yes").astype(int)
    X = df.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    return X, y


def get_train_test_split(
    df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Pipeline completo: load → split features/target → split estratificado.

    Returns: (X_train, X_test, y_train, y_test)
    """
    if df is None:
        df = load_clean_data()

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    logger.info(
        "train_test_split",
        train_shape=X_train.shape,
        test_shape=X_test.shape,
        train_churn_rate=float(y_train.mean()),
        test_churn_rate=float(y_test.mean()),
    )

    return X_train, X_test, y_train, y_test
