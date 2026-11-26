from __future__ import annotations

import itertools
import json

import pytest
from rapidfuzz.distance import DamerauLevenshtein

from travel_agent.config import CONFIG_DIR, read_yaml
from travel_agent.destinations import load_catalog
from travel_agent.lookups import (
    FUZZY_MIN_LENGTH,
    FUZZY_MIN_SIMILARITY,
    LOOKUPS_DIR,
    build_cities,
    build_currencies,
    load_lookups,
)

LOOKUPS = load_lookups()


def test_lookup_files_are_in_sync_with_configs() -> None:
    cities = json.loads((LOOKUPS_DIR / "cities.json").read_text("utf-8"))
    currencies = json.loads((LOOKUPS_DIR / "currencies.json").read_text("utf-8"))
    assert cities == build_cities(load_catalog()), "run scripts/build_lookups.py"
    assert currencies == build_currencies(read_yaml(CONFIG_DIR / "currencies.yaml"))


def test_no_two_cities_are_fuzzy_neighbours() -> None:
    aliases = LOOKUPS._aliases["city"]
    long_aliases = [a for a in aliases if len(a) >= FUZZY_MIN_LENGTH]
    for a, b in itertools.combinations(long_aliases, 2):
        if aliases[a] != aliases[b]:
            assert DamerauLevenshtein.normalized_similarity(a, b) < FUZZY_MIN_SIMILARITY, (a, b)


@pytest.mark.parametrize(
    ("raw", "city"),
    [
        ("Lisbonne", "lisbon"),
        ("LISBON", "lisbon"),
        ("au Caire", "cairo"),
        ("Caire", "cairo"),
        ("الدار البيضاء", "casablanca"),
        ("بباريس", "paris"),
        ("عمان", "amman"),
        ("جربه", "djerba"),
        ("nyc", "new_york"),
        ("Montreal", "montreal"),
        ("edinburg", "edinburgh"),  # typo
        ("Marakech", "marrakech"),  # typo
        ("barcelonne", "barcelona"),  # typo
        ("Tokio", "tokyo"),  # typo
    ],
)
def test_resolve_city(raw: str, city: str) -> None:
    match = LOOKUPS.resolve_place(raw)
    assert match is not None and match.id == city


@pytest.mark.parametrize("raw", ["Reykjavik", "Lyon", "Sydney", "مسقط", "Rio"])
def test_out_of_catalog_city(raw: str) -> None:
    assert LOOKUPS.resolve_place(raw) is None


@pytest.mark.parametrize(
    ("raw", "country"),
    [("Japon", "JP"), ("aux États-Unis", "US"), ("اليابان", "JP"), ("uk", "GB"), ("Maroc", "MA")],
)
def test_resolve_country(raw: str, country: str) -> None:
    match = LOOKUPS.resolve_place(raw, ("country",))
    assert match is not None and match.id == country


@pytest.mark.parametrize(
    ("raw", "iso"),
    [
        ("dinars", "TND"),  # documented default
        ("dinar algérien", "DZD"),
        ("دينار جزائري", "DZD"),
        ("dinars jordaniens", "JOD"),
        ("euros", "EUR"),
        ("€", "EUR"),
        ("بالدولار", "USD"),
        ("dollars", "USD"),
        ("dollars canadiens", "CAD"),
        ("درهم مغربي", "MAD"),
        ("dirham", "MAD"),
        ("dirhams des émirats", "AED"),
        ("livres turques", "TRY"),
        ("pounds", "GBP"),
        ("livre", "GBP"),
        ("جنيه", "EGP"),
        ("جنيه استرليني", "GBP"),
        ("yens", "JPY"),
        ("usd", "USD"),
        ("riyal qatari", "QAR"),
        ("ريال", "SAR"),
        ("franc suisse", "CHF"),
        ("دراهم", "MAD"),
    ],
)
def test_resolve_currency(raw: str, iso: str) -> None:
    assert LOOKUPS.resolve_currency(raw) == iso


@pytest.mark.parametrize("raw", ["bitcoin", "", "argent"])
def test_unknown_currency(raw: str) -> None:
    assert LOOKUPS.resolve_currency(raw) is None


def test_catalog_currencies_are_known_codes() -> None:
    codes = set(json.loads((LOOKUPS_DIR / "currencies.json").read_text("utf-8"))["codes"])
    assert load_catalog().currencies <= codes
