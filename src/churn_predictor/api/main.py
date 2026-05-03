"""FastAPI app for churn prediction."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import RedirectResponse

from churn_predictor.api.inference import (
    get_state,
    load_model_artifacts,
    predict as run_inference,
)

from churn_predictor.api.schemas import (
    PredictionResponse,
    HealthResponse,
    CustomerFeatures,
)

from churn_predictor.api.middleware import (
    latency_middleware,
    request_id_middleware,
)

from churn_predictor.utils.logging import get_logger

log = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan do FastAPI para carregar o modelo na inicialização da API."""
    log.info("Starting API and loading model artifacts...")
    
    try:
        load_model_artifacts()
        log.info("Model artifacts loaded successfully.")
    except FileNotFoundError as e:
        log.error("Model artifacts not found. API will start without model.", error=str(e))
    yield
    log.info("Shutting down API...")

app = FastAPI(
    title="Churn Predictor API",
    description="API para predição de churn usando um modelo de machine learning treinado.",
    version="1.0.0",
    lifespan=lifespan,
)

# Registrando middlewares
app.middleware("http")(latency_middleware)
app.middleware("http")(request_id_middleware)

@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redireciona pra documentação interativa."""
    return RedirectResponse(url="/docs")

@app.get("/health", response_model=HealthResponse, tags=["Health"], summary="Verifica a saúde da API e status do modelo.")
async def health() -> HealthResponse:
    """Endpoint de saúde da API. Sempre retorna http 200. Se o modelo não carregou, status sera 'degraded'."""
    try:
        state = get_state()
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_version=state.model_version,
            threshold=state.threshold,
        )
    except RuntimeError:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            model_version=None,
            threshold=None
        )
    
@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["inference"],
    summary="Predição de churn para 1 cliente",
    responses={
        503: {"description": "Modelo não disponível"},
        422: {"description": "Payload inválido (validação Pydantic)"},
    },
)
async def predict(features: CustomerFeatures, request: Request) -> PredictionResponse:
    """Recebe features de 1 cliente, retorna probabilidade e predição binária."""
    try:
        state = get_state()
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        ) from e

    t0 = time.perf_counter()
    result = run_inference(features.model_dump(), state=state)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    request_id = getattr(request.state, "request_id", "unknown")

    log.info(
        "prediction_made",
        request_id=request_id,
        churn_probability=round(result["churn_probability"], 4),
        churn_prediction=result["churn_prediction"],
        latency_ms=round(elapsed_ms, 2),
        model_version=result["model_version"],
    )

    return PredictionResponse(
        request_id=request_id,
        churn_probability=result["churn_probability"],
        churn_prediction=result["churn_prediction"],
        threshold=result["threshold"],
        model_version=result["model_version"],
        latency_ms=elapsed_ms,
    )