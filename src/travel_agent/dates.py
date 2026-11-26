"""Normalize raw date spans (fr/en/ar) to ISO dates, and durations to end dates.

Pipeline: fold the span -> rewrite fr/en/ar date words into canonical English (date_lexicon) ->
explicit rules for the forms dateparser misses or gets wrong (relative offsets, weekdays,
weekends, mid/early/end of month, day + month) -> dateparser as a fallback.

Conventions (documented in docs/annotation_guide.md):
- every expression resolves to a date strictly in the future of, or equal to, the reference date;
- "<weekday>", "this <weekday>" and "next <weekday>" = the first such weekday after the base date;
- "this weekend" = the coming Saturday (the base date itself if it is a Saturday or Sunday);
  "next weekend" = one week later;
- "next week" = base + 7 days; "next month" = first day of next month;
- "early <month>" = 1st, "mid <month>" = 15th, "end of <month>" = last day; a bare month = 1st;
- numeric dates are day-first for fr/ar and month-first for en;
- durations: N nights or N days -> end = start + N days; N weeks -> 7N days; N months -> 30N days.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Literal

import dateparser

from travel_agent.date_lexicon import MONTHS_EN, STOPWORDS, UNITS, WEEKDAYS_EN, build_phrase_table
from travel_agent.text_norm import fold

Lang = Literal["fr", "en", "ar"]

_ORDINAL_SUFFIX = re.compile(r"\b(\d{1,2})(st|nd|rd|th|er|ere|eme|e)\b")
_NUMERIC_DATE = re.compile(r"^(\d{1,4})/(\d{1,2})(?:/(\d{2,4}))?$")
_UNIT_DAYS = {"day": 1, "night": 1, "week": 7, "month": 30}


@lru_cache(maxsize=1)
def _phrase_regex() -> tuple[re.Pattern[str], dict[str, str]]:
    table = build_phrase_table()
    keys = sorted(table, key=len, reverse=True)
    pattern = re.compile(r"(?<!\w)(" + "|".join(re.escape(k) for k in keys) + r")(?!\w)")
    return pattern, table


def canonical_tokens(span: str) -> list[str]:
    """Fold a span and rewrite it into canonical English tokens (stopwords removed)."""
    text = _ORDINAL_SUFFIX.sub(r"\1", fold(span))
    pattern, table = _phrase_regex()
    text = pattern.sub(lambda m: table[m.group(1)], text)
    raw = text.split()
    tokens: list[str] = []
    for i, tok in enumerate(raw):
        nxt = raw[i + 1] if i + 1 < len(raw) else ""
        if tok in {"a", "an"} and nxt in UNITS:
            tokens.append("1")
        elif tok == "sept":  # French 7 before a unit, otherwise September
            tokens.append("7" if nxt in UNITS else "september")
        elif tok not in STOPWORDS:
            tokens.append(tok)
    return tokens


def _add_months(day: date, months: int) -> date:
    month_index = day.month - 1 + months
    year, month = day.year + month_index // 12, month_index % 12 + 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def _next_weekday(base: date, weekday: int) -> date:
    return base + timedelta(days=(weekday - base.weekday() - 1) % 7 + 1)


def _future_date(base: date, month: int, day: int, year: int | None) -> date | None:
    try:
        if year is not None:
            return date(year if year > 100 else 2000 + year, month, day)
        candidate = date(base.year, month, day)
        return candidate if candidate >= base else date(base.year + 1, month, day)
    except ValueError:
        return None


def _month_part(base: date, month: int | None, part: str, year: int | None) -> date | None:
    if month is None:  # "end of the month" -> current month
        month, year = base.month, base.year
    last = calendar.monthrange(year or base.year, month)[1]
    day = {"early": 1, "mid": 15, "end": last}[part]
    result = _future_date(base.replace(day=1), month, day, year)
    if result is not None and result < base and year is None:
        result = _future_date(
            base, month, min(day, calendar.monthrange(base.year + 1, month)[1]), base.year + 1
        )
    return result


def _parse_tokens(tokens: list[str], lang: str, base: date) -> date | None:
    months = [MONTHS_EN.index(t) + 1 for t in tokens if t in MONTHS_EN]
    weekdays = [WEEKDAYS_EN.index(t) for t in tokens if t in WEEKDAYS_EN]
    numbers = [int(t) for t in tokens if t.isdigit()]
    units = [UNITS[t] for t in tokens if t in UNITS]
    year = next((n for n in numbers if n >= 1900), None)
    small = [n for n in numbers if n < 100]
    has_next = "next" in tokens

    if tokens == ["today"]:
        return base
    if tokens == ["tomorrow"]:
        return base + timedelta(days=1)
    if "weekend" in tokens:
        saturday = base if base.weekday() >= 5 else _next_weekday(base, 5)
        return saturday + timedelta(days=7) if has_next else saturday
    if weekdays and not months:
        return _next_weekday(base, weekdays[0])
    for part in ("early", "mid", "end"):
        if part in tokens and (months or "month" in units):
            return _month_part(base, months[0] if months else None, part, year)
    if months:
        if small and 1 <= small[0] <= 31:
            return _future_date(base, months[0], small[0], year)
        if not small:
            if year is None and months[0] == base.month:
                return base
            return _future_date(base.replace(day=1), months[0], 1, year)
        return None
    if units and not weekdays:
        count = small[0] if small else 1
        unit = units[0]
        if has_next and not small:  # "next week" / "next month"
            if unit == "month":
                return _add_months(base.replace(day=1), 1)
            return base + timedelta(days=7 if unit == "week" else 1)
        if "this" in tokens and not small:
            return base
        if "in" in tokens or small:
            if unit == "month":
                return _add_months(base, count)
            return base + timedelta(days=count * _UNIT_DAYS[unit])
    if len(tokens) == 1 and small and 1 <= small[0] <= 31:  # "le 12": next occurrence of day 12
        day = small[0]
        try:
            candidate = base.replace(day=day)
        except ValueError:
            candidate = None
        if candidate is not None and candidate >= base:
            return candidate
        nxt = _add_months(base.replace(day=1), 1)
        return _future_date(nxt, nxt.month, day, nxt.year)
    if len(tokens) == 1 and (match := _NUMERIC_DATE.match(tokens[0])):
        a, b, c = match.group(1), match.group(2), match.group(3)
        if len(a) == 4:  # ISO-like 2026/12/15
            return _future_date(base, int(b), int(c), int(a)) if c else None
        day, month = (int(b), int(a)) if lang == "en" else (int(a), int(b))
        return _future_date(base, month, day, int(c) if c else None) if month <= 12 else None
    return None


def parse_date(span: str, lang: str, ref: date, *, base: date | None = None) -> date | None:
    """Resolve a raw date span. `base` (default `ref`) anchors relative forms, e.g. a date_end
    "dimanche" is the first Sunday after date_start."""
    anchor = base or ref
    if not span or not span.strip():
        return None
    tokens = canonical_tokens(span)
    if not tokens:
        return None
    result = _parse_tokens(tokens, lang, anchor)
    if result is not None:
        return result
    settings = {
        "RELATIVE_BASE": datetime.combine(anchor, datetime.min.time()),
        "PREFER_DATES_FROM": "future",
        "PREFER_DAY_OF_MONTH": "first",
        "DATE_ORDER": "MDY" if lang == "en" else "DMY",
    }
    parsed = dateparser.parse(" ".join(tokens), languages=["en"], settings=settings)
    return parsed.date() if parsed else None


def parse_duration_days(span: str) -> int | None:
    """'3 nuits' -> 3, 'pour une semaine' -> 7, 'لمدة ثلاث ليال' -> 3, 'for 2 weeks' -> 14."""
    tokens = canonical_tokens(span)
    if not tokens or any(t in MONTHS_EN or t in WEEKDAYS_EN for t in tokens):
        return None
    if {"in", "next", "this", "today", "tomorrow", "weekend"} & set(tokens):
        return None
    units = [UNITS[t] for t in tokens if t in UNITS]
    if len(units) != 1:
        return None
    numbers = [int(t) for t in tokens if t.isdigit()]
    if len(numbers) > 1:
        return None
    return (numbers[0] if numbers else 1) * _UNIT_DAYS[units[0]]


@dataclass(frozen=True)
class DateRange:
    start: date | None
    end: date | None


def resolve_dates(date_start: str | None, date_end: str | None, lang: str, ref: date) -> DateRange:
    """Normalize a (date_start, date_end) pair. date_end may be a date or a duration."""
    start = parse_date(date_start, lang, ref) if date_start else None
    end: date | None = None
    if date_end:
        days = parse_duration_days(date_end)
        if days is not None:
            end = start + timedelta(days=days) if start else None
        else:
            end = parse_date(date_end, lang, ref, base=start or ref)
    return DateRange(start=start, end=end)
