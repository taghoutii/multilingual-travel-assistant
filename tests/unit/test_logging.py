from __future__ import annotations

import io
import json

import pytest

from travel_agent.logging_setup import get_logger, log_stage, request_context, setup_logging


def _lines(stream: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_json_line_with_request_id_and_fields() -> None:
    stream = io.StringIO()
    setup_logging("INFO", stream)
    logger = get_logger("test")
    with request_context("req-1"):
        logger.info("hello", extra={"stage": "unit", "model": "m", "tokens": {"total": 3}})
    logger.info("outside")
    first, second = _lines(stream)
    assert first["event"] == "hello"
    assert first["request_id"] == "req-1"
    assert first["stage"] == "unit"
    assert first["tokens"] == {"total": 3}
    assert second["request_id"] is None


def test_non_ascii_kept_readable() -> None:
    stream = io.StringIO()
    setup_logging("INFO", stream)
    get_logger("test").info("مرحبا")
    assert "مرحبا" in stream.getvalue()


def test_log_stage_ok_and_error() -> None:
    stream = io.StringIO()
    setup_logging("INFO", stream)
    logger = get_logger("test")
    with log_stage(logger, "parse") as extra:
        extra["model"] = "m"
    with pytest.raises(ValueError), log_stage(logger, "route"):
        raise ValueError("boom")
    ok, failed = _lines(stream)
    assert ok["stage"] == "parse"
    assert ok["status"] == "ok"
    assert ok["model"] == "m"
    assert isinstance(ok["latency_ms"], float)
    assert failed["stage"] == "route"
    assert failed["status"] == "error"
    assert failed["error"] == "ValueError: boom"


def test_setup_is_idempotent() -> None:
    setup_logging("INFO", io.StringIO())
    logger = setup_logging("INFO", io.StringIO())
    assert len(logger.handlers) == 1
