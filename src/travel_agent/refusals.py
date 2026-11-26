"""Fixed replies for early exits (configs/refusal_templates.yaml)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from travel_agent.config import CONFIG_DIR, read_yaml
from travel_agent.language_id import SUPPORTED, Lang

TEMPLATES_PATH = CONFIG_DIR / "refusal_templates.yaml"


class _OffTopic(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    intro: str = Field(min_length=1)
    topics: list[str] = Field(min_length=1)


class RefusalTemplates(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    off_topic: dict[Lang, _OffTopic]
    unsupported_language: str = Field(min_length=1)

    @model_validator(mode="after")
    def _all_languages(self) -> RefusalTemplates:
        missing = set(SUPPORTED) - set(self.off_topic)
        if missing:
            raise ValueError(f"off_topic: missing languages {sorted(missing)}")
        sizes = {lang: len(t.topics) for lang, t in self.off_topic.items()}
        if len(set(sizes.values())) != 1:
            raise ValueError(f"off_topic: topic lists differ in length {sizes}")
        return self

    def off_topic_reply(self, lang: Lang) -> str:
        template = self.off_topic[lang]
        return "\n".join([template.intro, *(f"- {topic}" for topic in template.topics)])


def load_refusals(path: Path = TEMPLATES_PATH) -> RefusalTemplates:
    return RefusalTemplates.model_validate(read_yaml(path))


@lru_cache(maxsize=1)
def get_refusals() -> RefusalTemplates:
    return load_refusals()
