"""Load runtime settings (configs/settings.yaml) and secrets (.env)."""

from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(os.environ.get("TRAVEL_AGENT_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_DIR = PROJECT_ROOT / "configs"
SETTINGS_PATH = CONFIG_DIR / "settings.yaml"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HandlerSettings(_Frozen):
    timeout_s: float = Field(gt=0)
    retries: int = Field(ge=0)


class LLMClientSettings(_Frozen):
    timeout_s: float = Field(gt=0)
    max_retries: int = Field(ge=0)  # for 429, 5xx and timeouts
    backoff_base_s: float = Field(ge=0)
    backoff_max_s: float = Field(ge=0)


class ProviderSettings(_Frozen):
    base_url: str
    api_key_env: str


class ModelSettings(_Frozen):
    provider: str
    model: str | None
    family: str
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)


class Settings(_Frozen):
    seed: int
    log_level: str = "INFO"
    max_sub_requests: int = Field(ge=1)
    eval_reference_date: date
    handlers: HandlerSettings
    llm_client: LLMClientSettings
    providers: dict[str, ProviderSettings]
    models: dict[str, ModelSettings]
    understanding_base_model: str | None
    embeddings_model: str
    language_id_model: str

    @model_validator(mode="after")
    def _check_models(self) -> Settings:
        for role, cfg in self.models.items():
            if cfg.provider not in self.providers:
                raise ValueError(f"models.{role}: unknown provider {cfg.provider!r}")
        for role in ("generation", "ragas_judge"):
            if role not in self.models:
                raise ValueError(f"models.{role} is required")
        if self.models["ragas_judge"].family == self.models["generation"].family:
            raise ValueError("models.ragas_judge must be a different model family than generation")
        return self


class Secrets(BaseSettings):
    """API keys, read from the environment or .env. Missing keys stay None."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    groq_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    openweather_api_key: SecretStr | None = None
    exchangerate_api_key: SecretStr | None = None
    hf_token: SecretStr | None = None

    def get(self, env_name: str) -> str | None:
        """Return the secret stored under an environment variable name, e.g. 'GROQ_API_KEY'."""
        value = getattr(self, env_name.lower(), None)
        return value.get_secret_value() if isinstance(value, SecretStr) else None


def read_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_settings(path: Path = SETTINGS_PATH) -> Settings:
    return Settings.model_validate(read_yaml(path))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


@lru_cache(maxsize=1)
def get_secrets() -> Secrets:
    return Secrets()
