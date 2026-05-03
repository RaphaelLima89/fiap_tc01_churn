"""Middlewares da API: request_id e medição de latência."""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response

from churn_predictor.utils.logging import get_logger

log = get_logger(__name__)


async def request_id_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Middleware para gerar um UUID para cada requisição e ecoar no header de resposta."""

    incoming = request.headers.get("X-Request-ID")
    request_id = incoming if incoming else str(uuid.uuid4())
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


async def latency_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Middleware para medir a latência de cada requisição e logar junto com o request_id."""

    start_time = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.2f}"
    request.state.latency_ms = elapsed_ms

    log.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        latency_ms=round(elapsed_ms, 2),
        request_id=getattr(request.state, "request_id", None),
    )
    return response
