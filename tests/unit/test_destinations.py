from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from travel_agent.config import read_yaml
from travel_agent.destinations import DESTINATIONS_PATH, load_catalog


def _write(tmp_path: Path, raw: dict[str, object]) -> Path:
    path = tmp_path / "destinations.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return path


def test_catalog_size() -> None:
    catalog = load_catalog()
    assert len(catalog.cities) == 32
    assert len(catalog.countries) == 16


def test_every_country_has_two_cities() -> None:
    catalog = load_catalog()
    assert all(len(catalog.cities_in(code)) == 2 for code in catalog.countries)


def test_tunisia_entries() -> None:
    catalog = load_catalog()
    assert catalog.cities["tunis"].iata == "TUN"
    assert catalog.cities["djerba"].iata == "DJE"
    assert catalog.cities["djerba"].currency == "TND"


def test_catalog_currencies() -> None:
    assert {"EUR", "MAD", "TND", "JPY"} <= load_catalog().currencies


def test_unknown_country_rejected(tmp_path: Path) -> None:
    raw = read_yaml(DESTINATIONS_PATH)
    raw["cities"]["paris"]["country"] = "ZZ"
    with pytest.raises(ValidationError, match="unknown country"):
        load_catalog(_write(tmp_path, raw))


def test_duplicate_iata_rejected(tmp_path: Path) -> None:
    raw = read_yaml(DESTINATIONS_PATH)
    raw["cities"]["nice"]["iata"] = "CDG"
    with pytest.raises(ValidationError, match="duplicate IATA"):
        load_catalog(_write(tmp_path, raw))


def test_duplicate_name_rejected(tmp_path: Path) -> None:
    raw = read_yaml(DESTINATIONS_PATH)
    raw["cities"]["nice"]["names"]["en"] = "paris"
    with pytest.raises(ValidationError, match="duplicate en city names"):
        load_catalog(_write(tmp_path, raw))


def test_missing_language_rejected(tmp_path: Path) -> None:
    raw = read_yaml(DESTINATIONS_PATH)
    del raw["cities"]["paris"]["names"]["ar"]
    with pytest.raises(ValidationError):
        load_catalog(_write(tmp_path, raw))
