from __future__ import annotations

import pytest
from pydantic import ValidationError

from travel_agent.schemas import Entities, HandlerResult, ParsedQuery, Subcategory


def _request(subcategory: str = "weather", **entities: object) -> dict[str, object]:
    return {"subcategory": subcategory, "entities": entities}


def test_parse_multi_request() -> None:
    query = ParsedQuery.model_validate(
        {
            "requests": [
                _request(
                    "flight_search", origin="Paris", destination="Lisbon", date_start="next Friday"
                ),
                _request(
                    "hotel_search",
                    destination="Lisbon",
                    date_start="next Friday",
                    date_end="Sunday",
                ),
            ]
        }
    )
    assert [r.subcategory for r in query.requests] == [
        Subcategory("flight_search"),
        Subcategory("hotel_search"),
    ]
    assert query.requests[1].entities.destination == "Lisbon"
    assert query.requests[0].entities.travelers is None  # defaults are applied downstream


def test_parse_from_json() -> None:
    raw = '{"requests": [{"subcategory": "food", "entities": {"destination": "مراكش"}}]}'
    assert ParsedQuery.model_validate_json(raw).requests[0].entities.destination == "مراكش"


def test_empty_requests_rejected() -> None:
    with pytest.raises(ValidationError):
        ParsedQuery.model_validate({"requests": []})


def test_more_than_max_requests_rejected() -> None:
    with pytest.raises(ValidationError, match="at most 3"):
        ParsedQuery.model_validate({"requests": [_request() for _ in range(4)]})


def test_unknown_subcategory_rejected() -> None:
    with pytest.raises(ValidationError):
        ParsedQuery.model_validate({"requests": [_request("car_rental")]})


def test_unknown_entity_rejected() -> None:
    with pytest.raises(ValidationError):
        Entities.model_validate({"city": "Paris"})


def test_invalid_travelers_rejected() -> None:
    with pytest.raises(ValidationError):
        Entities(travelers=0)


def test_handler_result_unsupported() -> None:
    result = HandlerResult(
        subcategory=Subcategory("weather"),
        status="unsupported",
        data={"requested": "Reykjavik"},
        source="catalog",
    )
    assert result.missing_slots == []


def test_handler_result_bad_status_rejected() -> None:
    with pytest.raises(ValidationError):
        HandlerResult.model_validate(
            {"subcategory": "weather", "status": "maybe", "data": {}, "source": "x"}
        )
