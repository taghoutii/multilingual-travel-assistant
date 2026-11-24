from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from travel_agent.config import read_yaml
from travel_agent.schemas import Subcategory
from travel_agent.taxonomy import TAXONOMY_PATH, load_taxonomy


def _write(tmp_path: Path, raw: dict[str, object]) -> Path:
    path = tmp_path / "taxonomy.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_fifteen_subcategories_five_categories() -> None:
    taxonomy = load_taxonomy()
    assert len(taxonomy.subcategories) == 15
    groups = taxonomy.by_category()
    assert len(groups) == 5
    assert all(len(names) == 3 for names in groups.values())


def test_enum_matches_taxonomy() -> None:
    assert [s.value for s in Subcategory] == list(load_taxonomy().subcategories)


def test_flight_search_spec() -> None:
    spec = load_taxonomy().subcategories["flight_search"]
    assert spec.category == "flights"
    assert spec.handler == "flights"
    assert spec.required_slots == ("origin", "destination", "date_start")
    assert spec.defaults == {"travelers": 1}


def test_category_is_derived() -> None:
    taxonomy = load_taxonomy()
    assert taxonomy.category_of("currency_exchange") == "practical"
    assert taxonomy.category_of("contact_human") == "agency"


def test_unknown_slot_rejected(tmp_path: Path) -> None:
    raw = read_yaml(TAXONOMY_PATH)
    raw["subcategories"]["weather"]["required_slots"] = ["city"]
    with pytest.raises(ValueError, match="unknown slots"):
        load_taxonomy(_write(tmp_path, raw))


def test_unknown_category_rejected(tmp_path: Path) -> None:
    raw = read_yaml(TAXONOMY_PATH)
    raw["subcategories"]["weather"]["category"] = "misc"
    with pytest.raises(ValidationError, match="unknown category"):
        load_taxonomy(_write(tmp_path, raw))


def test_unknown_handler_rejected(tmp_path: Path) -> None:
    raw = read_yaml(TAXONOMY_PATH)
    raw["subcategories"]["weather"]["handler"] = "magic"
    with pytest.raises(ValidationError, match="unknown handler"):
        load_taxonomy(_write(tmp_path, raw))


def test_default_must_be_optional_slot(tmp_path: Path) -> None:
    raw = read_yaml(TAXONOMY_PATH)
    raw["subcategories"]["weather"]["defaults"] = {"destination": "Paris"}
    with pytest.raises(ValidationError, match="defaults for non-optional"):
        load_taxonomy(_write(tmp_path, raw))
