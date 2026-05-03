"""Configurando o logging via structlog"""

from __future__ import annotations

import logging
import os

import structlog


def configure_logging(level: str = "INFO", json_logs: bool | None = None) -> None:
    """Configura o structlog

    Args:
        level: Nível mínimo (DEBUG, INFO, WARNING, ERROR).
        json_logs:  Se True, emite JSON(producao). Se False, render colorido (dev).
                    Se none, decie env var LOG_JSON.
    """

    if json_logs is None:
        json_logs = os.getenv("LOG_JSON", "false").lower() == "true"

    # Logging stdlib em paralelo
    logging.basicConfig(format="%(message)s", level=level)

    # timestamp, level, contexto bound
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if json_logs:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Retorna o logger nomeado"""
    return structlog.get_logger(name)
