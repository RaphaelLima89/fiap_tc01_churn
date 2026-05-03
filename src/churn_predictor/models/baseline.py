"""Modelos baseline: Regressão Logística e Random Forest."""
from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from churn_predictor.config import RANDOM_SEED
from churn_predictor.features.preprocessing import build_preprocessor
from churn_predictor.utils.logging import get_logger

logger = get_logger(__name__)


def build_logreg() -> Pipeline:
    """Pipeline: preprocessor + LogReg com class_weight='balanced'."""
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_SEED,
        )),
    ])


def build_random_forest() -> Pipeline:
    """Pipeline: preprocessor + Random Forest com class_weight='balanced'."""
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )),
    ])


def evaluate(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    name: str,
) -> dict[str, Any]:
    """Avalia o modelo treinado no test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "f1": float(f1_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "report": classification_report(y_test, y_pred, output_dict=True),
    }

    logger.info(
        "model_evaluated",
        model=name,
        roc_auc=round(metrics["roc_auc"], 4),
        f1=round(metrics["f1"], 4),
    )
    return metrics