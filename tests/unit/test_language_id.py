"""Decision rule of the language identifier, with a fake fastText predictor (no model file)."""

from __future__ import annotations

import pytest

from travel_agent.language_id import LanguageIdentifier

PREDICTIONS: dict[str, list[tuple[str, float]]] = {
    "je veux un vol": [("fr", 0.96), ("en", 0.01)],
    "a flight please": [("en", 0.9), ("fr", 0.02)],
    "اريد رحلة": [("arz", 0.5), ("ar", 0.4), ("fa", 0.05)],
    "استرداد حجز الفندق": [("fa", 0.54), ("ar", 0.45)],
    "plats typiques djerba": [("sv", 0.47), ("fr", 0.06)],
    "vuelo madrid paris": [("es", 0.93), ("fr", 0.03)],
    "Wie ist das Wetter": [("de", 1.0)],
    "visa tokyo": [("en", 0.36), ("fr", 0.10)],
    "xyz": [("eo", 0.3)],
}


def fake_predictor(text: str) -> list[tuple[str, float]]:
    return PREDICTIONS.get(text, [])


@pytest.fixture
def lid() -> LanguageIdentifier:
    return LanguageIdentifier(fake_predictor, unsupported_min_prob=0.85, keep_previous_below=0.5)


@pytest.mark.parametrize(
    ("text", "lang", "reason"),
    [
        ("je veux un vol", "fr", "top"),
        ("a flight please", "en", "top"),
        ("اريد رحلة", "ar", "top"),  # Arabic varieties are merged: 0.5 + 0.4
        ("استرداد حجز الفندق", "ar", "best_supported"),  # fa below the unsupported threshold
        ("plats typiques djerba", "fr", "best_supported"),
    ],
)
def test_supported(lid: LanguageIdentifier, text: str, lang: str, reason: str) -> None:
    result = lid.identify(text)
    assert (result.lang, result.supported, result.reason) == (lang, True, reason)


@pytest.mark.parametrize("text", ["vuelo madrid paris", "Wie ist das Wetter"])
def test_unsupported_replies_in_english(lid: LanguageIdentifier, text: str) -> None:
    result = lid.identify(text)
    assert (result.lang, result.supported, result.reason) == ("en", False, "unsupported")
    assert result.detected in {"es", "de"}


def test_whitespace_is_collapsed_before_prediction(lid: LanguageIdentifier) -> None:
    assert lid.identify("  je veux\nun   vol ").lang == "fr"


@pytest.mark.parametrize(
    ("text", "lang"), [("vite", "en"), ("مرحبا", "ar"), ("", "en"), ("xyz", "en")]
)
def test_fallback_without_prediction(lid: LanguageIdentifier, text: str, lang: str) -> None:
    result = lid.identify(text)
    assert (result.lang, result.supported, result.reason) == (lang, True, "fallback")


def test_fallback_uses_previous_language(lid: LanguageIdentifier) -> None:
    assert lid.identify("vite", previous="fr").lang == "fr"
    assert lid.identify("xyz", previous="ar").reason == "previous"


def test_weak_detection_keeps_previous_language(lid: LanguageIdentifier) -> None:
    result = lid.identify("visa tokyo", previous="fr")
    assert (result.lang, result.reason) == ("fr", "previous")
    assert result.confidence == pytest.approx(0.10)


def test_strong_detection_switches_language(lid: LanguageIdentifier) -> None:
    assert lid.identify("a flight please", previous="fr").lang == "en"


def test_unsupported_wins_over_previous(lid: LanguageIdentifier) -> None:
    assert not lid.identify("Wie ist das Wetter", previous="fr").supported
