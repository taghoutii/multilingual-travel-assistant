"""Build the fictional mock data (data/mocks/flights.json, hotels.json) for the catalog cities.

Deterministic (seeded). Airlines and hotels are invented for the fictional Demo Travel Agency;
prices are illustrative, not market data. Flights are priced at query time by the flights handler
from these parameters and the great-circle distance between the two cities.
"""

from __future__ import annotations

import json
import random
import sys
from typing import Any

from travel_agent.config import PROJECT_ROOT, get_settings
from travel_agent.destinations import load_catalog

MOCKS_DIR = PROJECT_ROOT / "data" / "mocks"

# Relative cost of a hotel night (1.0 = average), set by hand for plausibility.
COST_INDEX = {
    "paris": 1.6,
    "nice": 1.3,
    "madrid": 1.1,
    "barcelona": 1.25,
    "rome": 1.3,
    "milan": 1.35,
    "lisbon": 1.0,
    "porto": 0.9,
    "london": 1.7,
    "edinburgh": 1.3,
    "berlin": 1.1,
    "munich": 1.3,
    "marrakech": 0.75,
    "casablanca": 0.7,
    "cairo": 0.55,
    "luxor": 0.5,
    "amman": 0.7,
    "aqaba": 0.75,
    "istanbul": 0.8,
    "antalya": 0.75,
    "dubai": 1.4,
    "abu_dhabi": 1.3,
    "new_york": 1.9,
    "los_angeles": 1.6,
    "montreal": 1.1,
    "toronto": 1.2,
    "tokyo": 1.4,
    "osaka": 1.15,
    "bangkok": 0.6,
    "phuket": 0.7,
    "tunis": 0.55,
    "djerba": 0.6,
}

AIRLINES = [
    {
        "code": "DA",
        "name": "Demo Air",
        "base_fare_eur": 45,
        "price_per_km_eur": 0.085,
        "cabin_bag_kg": 8,
        "checked_bags_included": 1,
        "checked_bag_kg": 23,
    },
    {
        "code": "SK",
        "name": "Skylark Airways",
        "base_fare_eur": 30,
        "price_per_km_eur": 0.075,
        "cabin_bag_kg": 7,
        "checked_bags_included": 0,
        "checked_bag_kg": 20,
    },
    {
        "code": "MW",
        "name": "Meridian Wings",
        "base_fare_eur": 60,
        "price_per_km_eur": 0.095,
        "cabin_bag_kg": 10,
        "checked_bags_included": 2,
        "checked_bag_kg": 23,
    },
    {
        "code": "AJ",
        "name": "Azure Jet",
        "base_fare_eur": 40,
        "price_per_km_eur": 0.08,
        "cabin_bag_kg": 8,
        "checked_bags_included": 1,
        "checked_bag_kg": 20,
    },
]

HOTEL_WORDS = [
    "Azure",
    "Cedar",
    "Harbor",
    "Olive",
    "Lantern",
    "Saffron",
    "Courtyard",
    "Meridian",
    "Atlas",
    "Jasmine",
    "Palm",
    "Riverside",
    "Old Town",
    "Garden",
    "Skyline",
    "Sandstone",
    "Marina",
    "Citadel",
    "Willow",
    "Coral",
    "Amber",
    "Terrace",
]
HOTEL_TYPES = ["Hotel", "Suites", "Inn", "Residence", "Boutique Hotel", "Lodge", "Palace"]
AMENITIES = [
    "wifi",
    "breakfast",
    "pool",
    "gym",
    "spa",
    "parking",
    "airport_shuttle",
    "restaurant",
    "family_rooms",
    "accessible_rooms",
    "air_conditioning",
    "pets_allowed",
]


def build_flights() -> dict[str, Any]:
    return {
        "note": "Fictional airlines and illustrative prices for the Demo Travel Agency.",
        "currency": "EUR",
        "airlines": AIRLINES,
        "departure_times": ["06:10", "08:45", "11:30", "14:20", "17:05", "20:40"],
        "cruise_speed_kmh": 780,
        "taxi_and_climb_minutes": 35,
        "max_direct_km": 6000,
        "connection_hubs": ["istanbul", "dubai", "paris", "london"],
        "connection_minutes": [75, 120, 165],
        "fare_classes": [
            {"name": "economy", "multiplier": 1.0},
            {"name": "premium_economy", "multiplier": 1.7},
        ],
        "weekend_multiplier": 1.15,
        "price_jitter": 0.12,
    }


def build_hotels(rng: random.Random) -> dict[str, Any]:
    catalog = load_catalog()
    hotels: list[dict[str, Any]] = []
    used: set[str] = set()
    for city_id, city in catalog.cities.items():
        for stars in (3, 4, 5):
            while True:
                name = f"{rng.choice(HOTEL_WORDS)} {rng.choice(HOTEL_TYPES)} {city.names.en}"
                if name not in used:
                    used.add(name)
                    break
            base = {3: 70, 4: 115, 5: 210}[stars] * COST_INDEX[city_id]
            amenities = sorted(rng.sample(AMENITIES, k=3 + stars - 2))
            hotels.append(
                {
                    "id": f"{city_id}-{stars}",
                    "city": city_id,
                    "name": name,
                    "stars": stars,
                    "price_per_night_eur": round(base * rng.uniform(0.9, 1.1)),
                    "rating": round(rng.uniform(7.2, 9.4), 1),
                    "distance_center_km": round(rng.uniform(0.3, 6.0), 1),
                    "breakfast_included": "breakfast" in amenities,
                    "amenities": amenities,
                    "free_cancellation_days": rng.choice([1, 2, 3, 7]),
                }
            )
    return {
        "note": "Fictional hotels and illustrative prices for the Demo Travel Agency.",
        "currency": "EUR",
        "hotels": hotels,
    }


def main() -> int:
    rng = random.Random(get_settings().seed)
    MOCKS_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in (("flights.json", build_flights()), ("hotels.json", build_hotels(rng))):
        path = MOCKS_DIR / name
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", "utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
