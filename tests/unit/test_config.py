from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from travel_agent.config import SETTINGS_PATH, Secrets, load_settings, read_yaml


def _write_settings(tmp_path: Path, **changes: object) -> Path:
    raw = read_yaml(SETTINGS_PATH)
    for dotted, value in changes.items():
        node = raw
        *parents, leaf = dotted.split("__")
        for key in parents:
            node = node[key]
        node[leaf] = value
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_repo_settings_load() -> None:
    settings = load_settings()
    assert settings.max_sub_requests == 3
    assert settings.eval_reference_date == date(2026, 12, 1)
    assert settings.handlers.timeout_s == 5.0
    assert settings.handlers.retries == 1
    generation = settings.models["generation"]
    assert generation.model == "openai/gpt-oss-120b"
    assert generation.provider == "groq"
    assert generation.temperature == 0.2


def test_judge_family_must_differ_from_generator(tmp_path: Path) -> None:
    path = _write_settings(tmp_path, models__ragas_judge__family="gpt-oss")
    with pytest.raises(ValidationError, match="different model family"):
        load_settings(path)


def test_unknown_provider_rejected(tmp_path: Path) -> None:
    path = _write_settings(tmp_path, models__generation__provider="nope")
    with pytest.raises(ValidationError, match="unknown provider"):
        load_settings(path)


def test_unknown_key_rejected(tmp_path: Path) -> None:
    path = _write_settings(tmp_path, typo_key=1)
    with pytest.raises(ValidationError):
        load_settings(path)


def test_secrets_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    secrets = Secrets(_env_file=None)
    assert secrets.get("GROQ_API_KEY") == "test-key"
    assert secrets.get("GEMINI_API_KEY") is None
    assert "test-key" not in repr(secrets)
