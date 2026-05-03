"""Pipeline de pré-processamento: scaling + encoding."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_predictor.data.loader import (
    BINARY_CAT_FEATURES,
    MULTI_CAT_FEATURES,
    NUMERIC_FEATURES,
)


def build_preprocessor() -> ColumnTransformer:
    """Pipeline sklearn para transformar X em matriz numérica.

    - Numéricas (3): StandardScaler.
    - Cat. binárias (13): OneHotEncoder(drop="if_binary") -> 1 coluna por feature.
    - Cat. multi-classe (3): OneHotEncoder(drop="first") -> (n-1) colunas.

    Total esperado: 3 + 13 + (2+2+3) = 23 features.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "bin",
                OneHotEncoder(
                    drop="if_binary",
                    sparse_output=False,
                    dtype="int8",
                    handle_unknown="ignore",
                ),
                BINARY_CAT_FEATURES,
            ),
            (
                "multi",
                OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    dtype="int8",
                    handle_unknown="ignore",
                ),
                MULTI_CAT_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
