"""Smoke test: pipeline de dados de ponta a ponta."""


from __future__ import annotations  

import pandas as pd

from churn_predictor.config import RANDOM_SEED, TARGET_COLUMN, TEST_SIZE
from churn_predictor.data.loader import get_train_test_split, load_clean_data

def test_train_test_split_works(clean_dataset_path) -> None:
    """Carrega o parquet limpo, faz split estratificado e valida invariantes."""
    # Carga
    df = load_clean_data()
    assert isinstance(df, pd.DataFrame)
    assert TARGET_COLUMN in df.columns
    assert len(df) > 0

    # Split estratificado completo
    X_train, X_test, y_train, y_test = get_train_test_split(df)

    # Shapes coerentes
    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)
    assert X_train.shape[1] == X_test.shape[1]

    # Test set com proporcao certa (TEST_SIZE = 0.2)
    expected_test_size = int(len(df) * TEST_SIZE)
    assert abs(len(X_test) - expected_test_size) <= 1

    # Estratificacao preservada (churn rate +/- 0.5pp entre train e test)
    assert abs(y_train.mean() - y_test.mean()) < 0.005

    # Reprodutibilidade: mesmo seed -> mesmo split
    X_train2, X_test2, _, _ = get_train_test_split(df)
    assert X_train.index.equals(X_train2.index)
    assert X_test.index.equals(X_test2.index)