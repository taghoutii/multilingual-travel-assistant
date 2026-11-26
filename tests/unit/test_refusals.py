from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from travel_agent.config import read_yaml
from travel_agent.refusals import TEMPLATES_PATH, load_refusals

REFUSALS = load_refusals()

EN_TOPICS = [
    "flights",
    "hotels",
    "weather",
    "currency exchange",
    "visas and travel documents",
    "travel insurance",
    "baggage",
    "flight changes and cancellations",
    "booking and payment information",
    "local food and attractions",
    "contacting an agent",
]


def test_english_lists_every_topic() -> None:
    assert REFUSALS.off_topic["en"].topics == EN_TOPICS


@pytest.mark.parametrize("lang", ["fr", "en", "ar"])
def test_reply_lists_topics_as_bullets(lang: str) -> None:
    reply = REFUSALS.off_topic_reply(lang)  # type: ignore[arg-type]
    lines = reply.splitlines()
    assert lines[0] == REFUSALS.off_topic[lang].intro  # type: ignore[index]
    assert len(lines) == 1 + len(EN_TOPICS)
    assert all(line.startswith("- ") for line in lines[1:])


def test_french_and_arabic_wording() -> None:
    assert "les vols" in REFUSALS.off_topic_reply("fr")
    assert "conseiller" in REFUSALS.off_topic_reply("fr")
    assert "الفنادق" in REFUSALS.off_topic_reply("ar")
    assert "تأمين السفر" in REFUSALS.off_topic_reply("ar")


def test_unsupported_language_notice_is_one_line() -> None:
    assert "\n" not in REFUSALS.unsupported_language
    assert "English" in REFUSALS.unsupported_language


def test_topic_lists_must_have_the_same_length(tmp_path: Path) -> None:
    raw = read_yaml(TEMPLATES_PATH)
    raw["off_topic"]["fr"]["topics"].pop()
    path = tmp_path / "t.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValidationError, match="differ in length"):
        load_refusals(path)


def test_every_language_is_required(tmp_path: Path) -> None:
    raw = read_yaml(TEMPLATES_PATH)
    del raw["off_topic"]["ar"]
    path = tmp_path / "t.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValidationError, match="missing languages"):
        load_refusals(path)
