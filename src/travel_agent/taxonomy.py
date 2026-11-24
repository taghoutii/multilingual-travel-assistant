"""Load and validate the taxonomy (configs/taxonomy.yaml)."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from travel_agent.config import CONFIG_DIR, read_yaml

TAXONOMY_PATH = CONFIG_DIR / "taxonomy.yaml"

SlotValue = int | float | str


class SubcategorySpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    category: str
    handler: str
    required_slots: tuple[str, ...] = ()
    optional_slots: tuple[str, ...] = ()
    defaults: dict[str, SlotValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_slots(self) -> SubcategorySpec:
        overlap = set(self.required_slots) & set(self.optional_slots)
        if overlap:
            raise ValueError(f"slots both required and optional: {sorted(overlap)}")
        unknown_defaults = set(self.defaults) - set(self.optional_slots)
        if unknown_defaults:
            raise ValueError(f"defaults for non-optional slots: {sorted(unknown_defaults)}")
        return self

    @property
    def slots(self) -> tuple[str, ...]:
        return self.required_slots + self.optional_slots


class Taxonomy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    categories: tuple[str, ...]
    handlers: tuple[str, ...]
    subcategories: dict[str, SubcategorySpec]

    @model_validator(mode="after")
    def _check_references(self) -> Taxonomy:
        for name, spec in self.subcategories.items():
            if spec.category not in self.categories:
                raise ValueError(f"{name}: unknown category {spec.category!r}")
            if spec.handler not in self.handlers:
                raise ValueError(f"{name}: unknown handler {spec.handler!r}")
        unused = set(self.categories) - {s.category for s in self.subcategories.values()}
        if unused:
            raise ValueError(f"categories without subcategories: {sorted(unused)}")
        return self

    def check_slot_names(self, entity_fields: Iterable[str]) -> None:
        """Raise ValueError if a slot is not a field of the Entities schema."""
        allowed = set(entity_fields)
        for name, spec in self.subcategories.items():
            unknown = set(spec.slots) - allowed
            if unknown:
                raise ValueError(f"{name}: unknown slots {sorted(unknown)}")

    def category_of(self, subcategory: str) -> str:
        return self.subcategories[subcategory].category

    def by_category(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {c: [] for c in self.categories}
        for name, spec in self.subcategories.items():
            out[spec.category].append(name)
        return out


def read_subcategory_names(path: Path = TAXONOMY_PATH) -> list[str]:
    """Subcategory names in file order, without full validation (used to build the enum)."""
    raw = read_yaml(path)
    return list(raw["subcategories"])


def load_taxonomy(path: Path = TAXONOMY_PATH) -> Taxonomy:
    """Load the taxonomy and check slot names against the Entities schema."""
    from travel_agent.schemas import Entities  # local: schemas builds its enum from this module

    taxonomy = Taxonomy.model_validate(read_yaml(path))
    taxonomy.check_slot_names(Entities.model_fields)
    return taxonomy


@lru_cache(maxsize=1)
def get_taxonomy() -> Taxonomy:
    return load_taxonomy()
