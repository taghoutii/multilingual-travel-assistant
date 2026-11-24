"""Data contracts shared by all pipeline stages."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from travel_agent.config import get_settings
from travel_agent.taxonomy import read_subcategory_names

if TYPE_CHECKING:
    Subcategory = StrEnum
else:
    # Built from configs/taxonomy.yaml so the taxonomy is never hard-coded.
    Subcategory = StrEnum("Subcategory", {name: name for name in read_subcategory_names()})


class Entities(BaseModel):
    """All optional; dates are raw text spans, normalized later in code."""

    model_config = ConfigDict(extra="forbid")

    origin: str | None = None
    destination: str | None = None
    date_start: str | None = None
    date_end: str | None = None
    travelers: int | None = Field(default=None, ge=1)
    budget_amount: float | None = Field(default=None, ge=0)
    budget_currency: str | None = None
    currency_from: str | None = None
    currency_to: str | None = None
    amount: float | None = Field(default=None, ge=0)


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subcategory: Subcategory
    entities: Entities = Field(default_factory=Entities)


class ParsedQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requests: list[Request] = Field(min_length=1)

    @field_validator("requests")
    @classmethod
    def _max_requests(cls, value: list[Request]) -> list[Request]:
        limit = get_settings().max_sub_requests
        if len(value) > limit:
            raise ValueError(f"at most {limit} requests per message, got {len(value)}")
        return value


HandlerStatus = Literal["ok", "missing_slots", "unavailable", "unsupported", "error"]


class HandlerResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subcategory: Subcategory
    status: HandlerStatus
    data: dict[str, Any]  # the facts the answer may use
    missing_slots: list[str] = Field(default_factory=list)
    source: str
