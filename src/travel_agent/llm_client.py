"""Async chat client for OpenAI-compatible endpoints (Groq, Gemini).

SDK retries are disabled. Retryable failures (429, 5xx, timeouts) are retried here with
exponential backoff, honoring Retry-After; other 4xx errors and connection errors are raised at
once. Retries and token usage are logged in one place.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

import httpx2
import openai
from openai import AsyncOpenAI

from travel_agent.config import Secrets, Settings, get_secrets, get_settings
from travel_agent.logging_setup import get_logger

logger = get_logger("llm_client")

Message = dict[str, str]
Sleep = Callable[[float], Awaitable[None]]


class LLMConfigError(RuntimeError):
    """The client cannot be built from settings (missing model or API key)."""


class LLMEmptyResponseError(RuntimeError):
    """The endpoint returned no message content."""


def _retry_reason(error: Exception) -> str | None:
    """Why an error is worth retrying ('429', '5xx', 'timeout'), or None if it is not."""
    if isinstance(error, openai.APITimeoutError):
        return "timeout"
    if isinstance(error, openai.APIStatusError):
        if error.status_code == 429:
            return "429"
        if error.status_code >= 500:
            return "5xx"
    return None


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class ChatResult:
    content: str
    model: str
    usage: TokenUsage
    latency_ms: float
    attempts: int


class LLMClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 20.0,
        http_client: httpx2.AsyncClient | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self._sleep = sleep
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_s,
            max_retries=0,
            http_client=http_client,
        )

    @classmethod
    def from_settings(
        cls,
        role: str,
        *,
        settings: Settings | None = None,
        secrets: Secrets | None = None,
        http_client: httpx2.AsyncClient | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> LLMClient:
        """Build the client for a role in settings.yaml (e.g. 'generation', 'ragas_judge')."""
        settings = settings or get_settings()
        secrets = secrets or get_secrets()
        if role not in settings.models:
            raise LLMConfigError(f"unknown model role {role!r}")
        model_cfg = settings.models[role]
        if model_cfg.model is None:
            raise LLMConfigError(f"models.{role}.model is not set in settings.yaml")
        provider = settings.providers[model_cfg.provider]
        api_key = secrets.get(provider.api_key_env)
        if not api_key:
            raise LLMConfigError(f"{provider.api_key_env} is not set (see .env.example)")
        retry = settings.llm_client
        return cls(
            base_url=provider.base_url,
            api_key=api_key,
            model=model_cfg.model,
            temperature=model_cfg.temperature,
            max_tokens=model_cfg.max_tokens,
            timeout_s=retry.timeout_s,
            max_retries=retry.max_retries,
            backoff_base_s=retry.backoff_base_s,
            backoff_max_s=retry.backoff_max_s,
            http_client=http_client,
            sleep=sleep,
        )

    def _backoff_s(self, attempt: int, error: Exception) -> float:
        retry_after = (
            error.response.headers.get("retry-after")
            if isinstance(error, openai.APIStatusError)
            else None
        )
        if retry_after is not None:
            try:
                return min(self.backoff_max_s, max(0.0, float(retry_after)))
            except ValueError:
                pass
        return min(self.backoff_max_s, self.backoff_base_s * 2**attempt)

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        response_format: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stage: str = "llm",
    ) -> ChatResult:
        """Send one chat completion; retry on 429, 5xx and timeouts. Logs model, tokens, latency."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature if temperature is None else temperature,
        }
        tokens_limit = self.max_tokens if max_tokens is None else max_tokens
        if tokens_limit is not None:
            kwargs["max_tokens"] = tokens_limit
        if response_format is not None:
            kwargs["response_format"] = response_format

        start = time.perf_counter()
        attempt = 0
        while True:
            try:
                completion = await self._client.chat.completions.create(**kwargs)
                break
            except openai.APIError as error:
                reason = _retry_reason(error)
                if reason is None or attempt >= self.max_retries:
                    logger.error(
                        "llm_failed",
                        extra={
                            "stage": stage,
                            "model": self.model,
                            "status": "error",
                            "attempt": attempt + 1,
                            "error": f"{type(error).__name__}: {reason or 'not retryable'}",
                        },
                    )
                    raise
                delay = self._backoff_s(attempt, error)
                logger.warning(
                    "llm_retry",
                    extra={
                        "stage": stage,
                        "model": self.model,
                        "status": "retry",
                        "attempt": attempt + 1,
                        "error": reason,
                        "latency_ms": round(delay * 1000),
                    },
                )
                await self._sleep(delay)
                attempt += 1

        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        usage = TokenUsage(
            prompt_tokens=completion.usage.prompt_tokens if completion.usage else 0,
            completion_tokens=completion.usage.completion_tokens if completion.usage else 0,
            total_tokens=completion.usage.total_tokens if completion.usage else 0,
        )
        content = completion.choices[0].message.content if completion.choices else None
        logger.info(
            "llm_call",
            extra={
                "stage": stage,
                "model": completion.model or self.model,
                "tokens": {
                    "prompt": usage.prompt_tokens,
                    "completion": usage.completion_tokens,
                    "total": usage.total_tokens,
                },
                "latency_ms": latency_ms,
                "attempt": attempt + 1,
                "status": "ok" if content else "error",
            },
        )
        if not content:
            raise LLMEmptyResponseError(f"empty response from {self.model}")
        return ChatResult(
            content=content,
            model=completion.model or self.model,
            usage=usage,
            latency_ms=latency_ms,
            attempts=attempt + 1,
        )

    async def aclose(self) -> None:
        await self._client.close()
