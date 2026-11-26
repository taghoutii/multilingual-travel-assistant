"""Committed data files: policies for every policy-handled subcategory, mocks over the catalog."""

from __future__ import annotations

import json
from collections import Counter

from travel_agent.config import PROJECT_ROOT
from travel_agent.destinations import load_catalog
from travel_agent.taxonomy import load_taxonomy

DATA = PROJECT_ROOT / "data"


def test_one_policy_doc_per_policy_subcategory() -> None:
    policy_subs = {n for n, s in load_taxonomy().subcategories.items() if s.handler == "policy"}
    docs = {p.stem for p in (DATA / "policies").glob("*.md")}
    assert docs == policy_subs


def test_policies_describe_the_fictional_agency() -> None:
    for path in (DATA / "policies").glob("*.md"):
        text = path.read_text("utf-8")
        assert "Demo Travel Agency (fictional)" in text or "This assistant" in text, path.name
    assert "embassy" in (DATA / "policies" / "visa_documents.md").read_text("utf-8")


def test_hotels_cover_every_catalog_city() -> None:
    hotels = json.loads((DATA / "mocks" / "hotels.json").read_text("utf-8"))["hotels"]
    per_city = Counter(h["city"] for h in hotels)
    assert set(per_city) == set(load_catalog().cities)
    assert set(per_city.values()) == {3}
    assert len({h["id"] for h in hotels}) == len(hotels)
    assert len({h["name"] for h in hotels}) == len(hotels)
    assert all(h["price_per_night_eur"] > 0 and h["stars"] in (3, 4, 5) for h in hotels)


def test_flight_parameters() -> None:
    flights = json.loads((DATA / "mocks" / "flights.json").read_text("utf-8"))
    codes = [a["code"] for a in flights["airlines"]]
    assert len(codes) == len(set(codes)) >= 3
    assert set(flights["connection_hubs"]) <= set(load_catalog().cities)
