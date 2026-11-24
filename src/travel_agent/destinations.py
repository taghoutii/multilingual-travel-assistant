"""Load and validate the destination catalog (configs/destinations.yaml)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from travel_agent.config import CONFIG_DIR, read_yaml

DESTINATIONS_PATH = CONFIG_DIR / "destinations.yaml"
LANGUAGES = ("fr", "en", "ar")


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Names(_Frozen):
    fr: str = Field(min_length=1)
    en: str = Field(min_length=1)
    ar: str = Field(min_length=1)

    def all(self) -> tuple[str, str, str]:
        return (self.fr, self.en, self.ar)


class Country(_Frozen):
    names: Names


class City(_Frozen):
    names: Names
    iata: str = Field(pattern=r"^[A-Z]{3}$")
    country: str = Field(pattern=r"^[A-Z]{2}$")
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class Catalog(_Frozen):
    countries: dict[str, Country]
    cities: dict[str, City]

    @model_validator(mode="after")
    def _check(self) -> Catalog:
        for city_id, city in self.cities.items():
            if city.country not in self.countries:
                raise ValueError(f"{city_id}: unknown country {city.country!r}")
        iatas = [c.iata for c in self.cities.values()]
        dup_iata = {i for i in iatas if iatas.count(i) > 1}
        if dup_iata:
            raise ValueError(f"duplicate IATA codes: {sorted(dup_iata)}")
        for lang in LANGUAGES:
            names = [getattr(c.names, lang).casefold() for c in self.cities.values()]
            dup = {n for n in names if names.count(n) > 1}
            if dup:
                raise ValueError(f"duplicate {lang} city names: {sorted(dup)}")
        unused = set(self.countries) - {c.country for c in self.cities.values()}
        if unused:
            raise ValueError(f"countries without cities: {sorted(unused)}")
        return self

    def cities_in(self, country: str) -> list[str]:
        return [cid for cid, c in self.cities.items() if c.country == country]

    @property
    def currencies(self) -> frozenset[str]:
        return frozenset(c.currency for c in self.cities.values())


def load_catalog(path: Path = DESTINATIONS_PATH) -> Catalog:
    return Catalog.model_validate(read_yaml(path))


@lru_cache(maxsize=1)
def get_catalog() -> Catalog:
    return load_catalog()
