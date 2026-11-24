"""LLM client tests: every HTTP call goes to an in-memory mock transport."""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from typing import Any

import httpx2
import openai
import pytest

from travel_agent.config import Secrets, load_settings
from travel_agent.llm_client import LLMClient, LLMConfigError, LLMEmptyResponseError
from travel_agent.logging_setup import setup_logging

Handler = Callable[[httpx2.Request], httpx2.Response]


def _completion(content: str | None = "bonjour") -> dict[str, Any]:
    return {
        "id": "cmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
    }


def _rate_limited(retry_after: str | None = None) -> httpx2.Response:
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    return httpx2.Response(429, headers=headers, json={"error": {"message": "slow down"}})


def _server_error(status: int = 500, retry_after: str | None = None) -> httpx2.Response:
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    return httpx2.Response(status, headers=headers, json={"error": {"message": "boom"}})


def _client_error(status: int) -> httpx2.Response:
    return httpx2.Response(status, json={"error": {"message": "bad request"}})


Outcome = httpx2.Response | type[httpx2.TransportError]


class Recorder:
    """Mock endpoint: returns (or raises, for transport errors) one outcome per request."""

    def __init__(self, outcomes: list[Outcome]) -> None:
        self.outcomes = outcomes
        self.requests: list[dict[str, Any]] = []
        self.sleeps: list[float] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, httpx2.Response):
            return outcome
        raise outcome("simulated", request=request)

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


def _client(recorder: Recorder, **overrides: Any) -> LLMClient:
    params: dict[str, Any] = {
        "base_url": "https://llm.test/v1",
        "api_key": "k",
        "model": "test-model",
        "temperature": 0.2,
        "backoff_base_s": 1.0,
        "backoff_max_s": 5.0,
        "http_client": httpx2.AsyncClient(transport=httpx2.MockTransport(recorder.handler)),
        "sleep": recorder.sleep,
    }
    params.update(overrides)
    return LLMClient(**params)


async def test_chat_returns_content_and_usage() -> None:
    recorder = Recorder([httpx2.Response(200, json=_completion())])
    result = await _client(recorder).chat([{"role": "user", "content": "salut"}])
    assert result.content == "bonjour"
    assert result.usage.total_tokens == 15
    assert result.attempts == 1
    sent = recorder.requests[0]
    assert sent["model"] == "test-model"
    assert sent["temperature"] == 0.2
    assert sent["messages"] == [{"role": "user", "content": "salut"}]


async def test_response_format_is_sent() -> None:
    recorder = Recorder([httpx2.Response(200, json=_completion("{}"))])
    fmt = {"type": "json_schema", "json_schema": {"name": "q", "schema": {"type": "object"}}}
    await _client(recorder).chat([{"role": "user", "content": "x"}], response_format=fmt)
    assert recorder.requests[0]["response_format"] == fmt


def _ok() -> httpx2.Response:
    return httpx2.Response(200, json=_completion())


async def _chat(recorder: Recorder) -> Any:
    return await _client(recorder).chat([{"role": "user", "content": "x"}])


def test_default_is_two_retries() -> None:
    assert _client(Recorder([])).max_retries == 2


# --- 429 -------------------------------------------------------------------------------------


async def test_429_retried_with_retry_after_then_backoff() -> None:
    recorder = Recorder([_rate_limited("3"), _rate_limited(), _ok()])
    result = await _chat(recorder)
    assert result.attempts == 3
    assert recorder.sleeps == [3.0, 2.0]  # Retry-After, then base * 2**1


async def test_429_gives_up_after_two_retries() -> None:
    recorder = Recorder([_rate_limited() for _ in range(3)])
    with pytest.raises(openai.RateLimitError):
        await _chat(recorder)
    assert len(recorder.requests) == 3
    assert recorder.sleeps == [1.0, 2.0]


async def test_backoff_is_capped() -> None:
    recorder = Recorder([_rate_limited("120"), _ok()])
    await _chat(recorder)
    assert recorder.sleeps == [5.0]


# --- 5xx -------------------------------------------------------------------------------------


@pytest.mark.parametrize("status", [500, 502, 503, 504])
async def test_5xx_retried_with_exponential_backoff(status: int) -> None:
    recorder = Recorder([_server_error(status), _server_error(status), _ok()])
    result = await _chat(recorder)
    assert result.attempts == 3
    assert recorder.sleeps == [1.0, 2.0]


async def test_5xx_honors_retry_after() -> None:
    recorder = Recorder([_server_error(503, retry_after="4"), _ok()])
    await _chat(recorder)
    assert recorder.sleeps == [4.0]


async def test_5xx_gives_up_after_two_retries() -> None:
    recorder = Recorder([_server_error() for _ in range(3)])
    with pytest.raises(openai.InternalServerError):
        await _chat(recorder)
    assert len(recorder.requests) == 3
    assert recorder.sleeps == [1.0, 2.0]


# --- timeouts --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "timeout", [httpx2.ReadTimeout, httpx2.ConnectTimeout, httpx2.WriteTimeout, httpx2.PoolTimeout]
)
async def test_timeout_retried(timeout: type[httpx2.TransportError]) -> None:
    recorder = Recorder([timeout, _ok()])
    result = await _chat(recorder)
    assert result.attempts == 2
    assert recorder.sleeps == [1.0]


async def test_timeout_gives_up_after_two_retries() -> None:
    recorder = Recorder([httpx2.ReadTimeout for _ in range(3)])
    with pytest.raises(openai.APITimeoutError):
        await _chat(recorder)
    assert len(recorder.requests) == 3
    assert recorder.sleeps == [1.0, 2.0]


async def test_mixed_failures_share_one_retry_budget() -> None:
    recorder = Recorder([_rate_limited(), _server_error(), httpx2.ReadTimeout])
    with pytest.raises(openai.APITimeoutError):
        await _chat(recorder)
    assert len(recorder.requests) == 3


# --- never retried ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422])
async def test_other_4xx_never_retried(status: int) -> None:
    recorder = Recorder([_client_error(status), _ok()])
    with pytest.raises(openai.APIStatusError) as excinfo:
        await _chat(recorder)
    assert excinfo.value.status_code == status
    assert len(recorder.requests) == 1
    assert recorder.sleeps == []


async def test_connection_error_not_retried() -> None:
    recorder = Recorder([httpx2.ConnectError, _ok()])
    with pytest.raises(openai.APIConnectionError) as excinfo:
        await _chat(recorder)
    assert not isinstance(excinfo.value, openai.APITimeoutError)
    assert len(recorder.requests) == 1


async def test_empty_content_raises() -> None:
    recorder = Recorder([httpx2.Response(200, json=_completion(None))])
    with pytest.raises(LLMEmptyResponseError):
        await _client(recorder).chat([{"role": "user", "content": "x"}])


async def test_token_usage_is_logged() -> None:
    stream = io.StringIO()
    setup_logging("INFO", stream)
    recorder = Recorder([_server_error(502), _ok()])
    await _client(recorder).chat([{"role": "user", "content": "x"}], stage="generation")
    lines = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [line["event"] for line in lines] == ["llm_retry", "llm_call"]
    assert lines[0]["error"] == "5xx"
    call = lines[-1]
    assert call["stage"] == "generation"
    assert call["model"] == "test-model"
    assert call["tokens"] == {"prompt": 11, "completion": 4, "total": 15}
    assert call["status"] == "ok"


def _secrets(monkeypatch: pytest.MonkeyPatch, **env: str) -> Secrets:
    for name in ("GROQ_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return Secrets(_env_file=None)


async def test_failure_is_logged() -> None:
    stream = io.StringIO()
    setup_logging("INFO", stream)
    with pytest.raises(openai.BadRequestError):
        await _chat(Recorder([_client_error(400)]))
    (line,) = [json.loads(raw) for raw in stream.getvalue().splitlines()]
    assert line["event"] == "llm_failed"
    assert line["status"] == "error"
    assert line["error"] == "BadRequestError: not retryable"


async def test_from_settings_uses_role_config(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder([httpx2.Response(200, json=_completion())])
    client = LLMClient.from_settings(
        "generation",
        settings=load_settings(),
        secrets=_secrets(monkeypatch, GROQ_API_KEY="groq-key"),
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(recorder.handler)),
        sleep=recorder.sleep,
    )
    await client.chat([{"role": "user", "content": "x"}])
    assert client.max_retries == 2
    assert recorder.requests[0]["model"] == "openai/gpt-oss-120b"
    assert recorder.requests[0]["temperature"] == 0.2


def test_from_settings_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(LLMConfigError, match="GROQ_API_KEY"):
        LLMClient.from_settings("generation", secrets=_secrets(monkeypatch))


def test_from_settings_unpinned_model(monkeypatch: pytest.MonkeyPatch) -> None:
    secrets = _secrets(monkeypatch, GEMINI_API_KEY="g")
    with pytest.raises(LLMConfigError, match="not set in settings"):
        LLMClient.from_settings("ragas_judge", secrets=secrets)
