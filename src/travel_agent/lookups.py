"""Place and currency lookups (data/lookups/*.json), with fuzzy matching for typos.

The JSON files are derived from configs/destinations.yaml and configs/currencies.yaml by
scripts/build_lookups.py; `build_*` below produce their content.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from rapidfuzz.distance import DamerauLevenshtein

from travel_agent.config import PROJECT_ROOT
from travel_agent.destinations import Catalog
from travel_agent.text_norm import fold

LOOKUPS_DIR = PROJECT_ROOT / "data" / "lookups"
PlaceKind = Literal["city", "country"]

FUZZY_MIN_SIMILARITY = 0.8  # normalized Damerau-Levenshtein: one edit on a 5-letter name
FUZZY_MIN_LENGTH = 5
_LEADING_WORDS = frozenset(
    fold(w)
    for w in "a à au aux en le la les l de du d des pour vers sur to in the from at for into "
    "الى إلى في من ل".split()
)
_ARABIC_ARTICLE = "ال"

# Common short forms that are not catalog names (kept small and explicit).
EXTRA_COUNTRY_ALIASES: dict[str, list[str]] = {
    "US": ["usa", "us", "etats unis", "amerique", "america", "امريكا"],
    "GB": [
        "uk",
        "angleterre",
        "england",
        "britain",
        "great britain",
        "grande bretagne",
        "بريطانيا",
        "انجلترا",
    ],
    "AE": ["uae", "emirats", "الامارات"],
}
EXTRA_CITY_ALIASES: dict[str, list[str]] = {"new_york": ["nyc", "ny"]}


def _strip_article(name: str) -> str:
    words = name.split()
    if len(words) > 1 and words[0] in _LEADING_WORDS:
        return " ".join(words[1:])
    if words and words[0].startswith(_ARABIC_ARTICLE) and len(words[0]) > 4:
        return " ".join([words[0][2:], *words[1:]])
    return name


def _aliases(names: list[str]) -> set[str]:
    out: set[str] = set()
    for name in names:
        folded = fold(name)
        out.add(folded)
        out.add(_strip_article(folded))
    return {a for a in out if a}


def build_cities(catalog: Catalog) -> dict[str, Any]:
    city_aliases: dict[str, str] = {}
    country_aliases: dict[str, str] = {}
    for city_id, city in catalog.cities.items():
        names = [*city.names.all(), *EXTRA_CITY_ALIASES.get(city_id, [])]
        for alias in _aliases(names) | {city.iata.lower()}:
            if city_aliases.setdefault(alias, city_id) != city_id:
                raise ValueError(f"city alias {alias!r} is ambiguous")
    for code, country in catalog.countries.items():
        for alias in _aliases([*country.names.all(), *EXTRA_COUNTRY_ALIASES.get(code, [])]):
            if country_aliases.setdefault(alias, code) != code:
                raise ValueError(f"country alias {alias!r} is ambiguous")
    return {
        "cities": {cid: c.model_dump() for cid, c in sorted(catalog.cities.items())},
        "countries": {code: c.model_dump() for code, c in sorted(catalog.countries.items())},
        "aliases": {
            "city": dict(sorted(city_aliases.items())),
            "country": dict(sorted(country_aliases.items())),
        },
    }


def build_currencies(raw: dict[str, Any]) -> dict[str, Any]:
    words: dict[str, str] = {}
    for iso, forms in raw["words"].items():
        for form in forms:
            words[fold(str(form))] = iso
    ambiguous = {
        name: {
            "default": group["default"],
            "words": sorted({fold(str(w)) for w in group["words"]}),
            "qualifiers": {
                fold(str(q)): iso for iso, qs in group["qualifiers"].items() for q in qs
            },
        }
        for name, group in raw["ambiguous"].items()
    }
    return {
        "codes": sorted(raw["codes"]),
        "words": dict(sorted(words.items())),
        "ambiguous": ambiguous,
        "arabic_plurals": {fold(k): fold(v) for k, v in raw["arabic_plurals"].items()},
    }


@dataclass(frozen=True)
class PlaceMatch:
    kind: PlaceKind
    id: str
    similarity: float  # 1.0 = exact alias match


class Lookups:
    def __init__(self, cities: dict[str, Any], currencies: dict[str, Any]) -> None:
        self.cities = cities["cities"]
        self.countries = cities["countries"]
        self._aliases: dict[PlaceKind, dict[str, str]] = {
            "city": cities["aliases"]["city"],
            "country": cities["aliases"]["country"],
        }
        self._cur = currencies

    # ---- places -----------------------------------------------------------------------------

    def resolve_place(
        self, raw: str, kinds: tuple[PlaceKind, ...] = ("city",)
    ) -> PlaceMatch | None:
        """Exact alias match first, then a fuzzy match (one or two edits) for typos."""
        query = fold(raw)
        candidates = [query, _strip_article(query)]
        if query[:1] in {"ب", "ل"} and len(query) > 4:  # attached Arabic preposition
            candidates.append(query[1:])
        for kind in kinds:
            for cand in candidates:
                if cand in self._aliases[kind]:
                    return PlaceMatch(kind, self._aliases[kind][cand], 1.0)
        best: PlaceMatch | None = None
        tie = False
        for kind in kinds:
            for cand in candidates:
                if len(cand) < FUZZY_MIN_LENGTH:
                    continue
                for alias, target in self._aliases[kind].items():
                    if len(alias) < FUZZY_MIN_LENGTH:
                        continue
                    sim = DamerauLevenshtein.normalized_similarity(cand, alias)
                    if sim < FUZZY_MIN_SIMILARITY:
                        continue
                    if best is None or sim > best.similarity:
                        best, tie = PlaceMatch(kind, target, sim), False
                    elif sim == best.similarity and target != best.id:
                        tie = True
        return None if tie else best

    # ---- currencies -------------------------------------------------------------------------

    def _token_forms(self, token: str) -> list[str]:
        forms = [token]
        plurals = self._cur["arabic_plurals"]
        for prefix in ("بال", "ال", "ب", "و"):
            if token.startswith(prefix) and len(token) > len(prefix) + 2:
                forms.append(token[len(prefix) :])
        if token.isascii() and len(token) > 3 and token[-1] in "sx":
            forms.append(token[:-1])
        return [plurals.get(f, f) for f in forms]

    def resolve_currency(self, raw: str) -> str | None:
        """'dinars' -> TND (default), 'dinar algérien' -> DZD, 'بالدولار' -> USD, 'eur' -> EUR."""
        folded = fold(raw)
        tokens = [t for t in folded.split() if t not in _LEADING_WORDS]
        if not tokens:
            return None
        codes = set(self._cur["codes"])
        if len(tokens) == 1 and tokens[0].upper() in codes:
            return str(tokens[0].upper())
        forms = [self._token_forms(t) for t in tokens]
        words: dict[str, str] = self._cur["words"]
        # longest contiguous phrase made of any token forms
        for size in range(len(forms), 0, -1):
            for start in range(len(forms) - size + 1):
                for phrase in _combinations(forms[start : start + size]):
                    if phrase in words:
                        return words[phrase]
        flat = {f for fs in forms for f in fs}
        for group in self._cur["ambiguous"].values():
            if flat & set(group["words"]):
                for form in flat:
                    if form in group["qualifiers"]:
                        return str(group["qualifiers"][form])
                return str(group["default"])
        return None


def _combinations(options: list[list[str]]) -> list[str]:
    phrases = [""]
    for opts in options:
        phrases = [f"{p} {o}".strip() for p in phrases for o in dict.fromkeys(opts)]
    return phrases


def load_lookups(directory: Path = LOOKUPS_DIR) -> Lookups:
    with (directory / "cities.json").open(encoding="utf-8") as f:
        cities = json.load(f)
    with (directory / "currencies.json").open(encoding="utf-8") as f:
        currencies = json.load(f)
    return Lookups(cities, currencies)


@lru_cache(maxsize=1)
def get_lookups() -> Lookups:
    return load_lookups()
