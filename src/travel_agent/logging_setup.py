"""Structured JSON logging: request_id, stage, latency_ms, model, tokens, status."""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import IO, Any

LOGGER_NAME = "travel_agent"
STRUCTURED_FIELDS = ("stage", "latency_ms", "model", "tokens", "status", "attempt", "error")

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        for field in STRUCTURED_FIELDS:
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: str = "INFO", stream: IO[str] | None = None) -> logging.Logger:
    """Configure the package logger with one JSON handler (idempotent)."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


@contextmanager
def request_context(request_id: str | None = None) -> Iterator[str]:
    """Bind a request_id to every log line emitted inside the block."""
    rid = request_id or uuid.uuid4().hex
    token = request_id_var.set(rid)
    try:
        yield rid
    finally:
        request_id_var.reset(token)


@contextmanager
def log_stage(logger: logging.Logger, stage: str) -> Iterator[dict[str, Any]]:
    """Log one line per stage with latency and status; the yielded dict adds extra fields."""
    extra: dict[str, Any] = {}
    start = time.perf_counter()
    try:
        yield extra
    except Exception as exc:
        latency = round((time.perf_counter() - start) * 1000, 1)
        logger.error(
            "stage_failed",
            extra={
                **extra,
                "stage": stage,
                "latency_ms": latency,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    latency = round((time.perf_counter() - start) * 1000, 1)
    logger.info(
        "stage_done",
        extra={"status": "ok", **extra, "stage": stage, "latency_ms": latency},
    )
