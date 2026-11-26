"""Every expression from the dateparser probes (ar, fr, en) is a test case.

Reference date: 2026-12-01 (a Tuesday), the eval reference date.
"""

from __future__ import annotations

from datetime import date

import pytest

from travel_agent.dates import canonical_tokens, parse_date, parse_duration_days, resolve_dates

REF = date(2026, 12, 1)

AR = [
    ("اليوم", "2026-12-01"),
    ("غدا", "2026-12-02"),
    ("بعد غد", "2026-12-03"),
    ("بعد 3 أيام", "2026-12-04"),
    ("بعد ثلاثة أيام", "2026-12-04"),
    ("بعد أسبوعين", "2026-12-15"),
    ("الأسبوع القادم", "2026-12-08"),
    ("الشهر القادم", "2027-01-01"),
    ("يوم الجمعة", "2026-12-04"),
    ("الجمعة القادمة", "2026-12-04"),
    ("يوم الاثنين المقبل", "2026-12-07"),
    ("15 ديسمبر", "2026-12-15"),
    ("15 كانون الأول", "2026-12-15"),
    ("٢٠ يناير ٢٠٢٧", "2027-01-20"),
    ("3 مارس", "2027-03-03"),
    ("الأول من يناير", "2027-01-01"),
    ("الخامس عشر من ديسمبر", "2026-12-15"),
    ("25/12/2026", "2026-12-25"),
    ("منتصف فبراير", "2027-02-15"),
    ("نهاية الأسبوع", "2026-12-05"),
    # variants found while probing, plus casual spelling (no hamza, ه for ة)
    ("الجمعة", "2026-12-04"),
    ("خلال 3 أيام", "2026-12-04"),
    ("في 3 أيام", "2026-12-04"),
    ("بعد يومين", "2026-12-03"),
    ("بعد أسبوع", "2026-12-08"),
    ("بعد شهر", "2027-01-01"),
    ("الجمعه القادمه", "2026-12-04"),
    ("الاسبوع الجاي", "2026-12-08"),
    ("نهاية الشهر", "2026-12-31"),
    ("أواخر يناير", "2027-01-31"),
    ("20 جانفي", "2027-01-20"),
    ("5 دجنبر", "2026-12-05"),
]

FR = [
    ("aujourd'hui", "2026-12-01"),
    ("demain", "2026-12-02"),
    ("apres demain", "2026-12-03"),  # dateparser alone silently returns 2026-12-02
    ("après-demain", "2026-12-03"),
    ("dans 3 jours", "2026-12-04"),
    ("dans trois jours", "2026-12-04"),
    ("dans deux semaines", "2026-12-15"),
    ("la semaine prochaine", "2026-12-08"),
    ("le mois prochain", "2027-01-01"),
    ("vendredi", "2026-12-04"),
    ("vendredi prochain", "2026-12-04"),
    ("ce weekend", "2026-12-05"),
    ("le week-end prochain", "2026-12-12"),
    ("15 décembre", "2026-12-15"),
    ("15 decembre", "2026-12-15"),
    ("le 15 décembre", "2026-12-15"),
    ("12/01", "2027-01-12"),
    ("1er janvier", "2027-01-01"),
    ("premier janvier", "2027-01-01"),
    ("mi-février", "2027-02-15"),
    ("fin janvier", "2027-01-31"),
    ("début mars", "2027-03-01"),
    ("mars", "2027-03-01"),
    ("dem", "2026-12-02"),
    # extra forms used in the synthetic data
    ("ce we", "2026-12-05"),
    ("le 7 sept", "2027-09-07"),
    ("vingt et un janvier", "2027-01-21"),
    ("le 12", "2026-12-12"),
    ("fin du mois", "2026-12-31"),
    ("samedi 19 décembre", "2026-12-19"),
]

EN = [
    ("today", "2026-12-01"),
    ("tomorrow", "2026-12-02"),
    ("tmrw", "2026-12-02"),
    ("day after tomorrow", "2026-12-03"),
    ("in 3 days", "2026-12-04"),
    ("in three days", "2026-12-04"),
    ("in two weeks", "2026-12-15"),
    ("next week", "2026-12-08"),
    ("next month", "2027-01-01"),
    ("friday", "2026-12-04"),
    ("next friday", "2026-12-04"),
    ("this weekend", "2026-12-05"),
    ("next weekend", "2026-12-12"),
    ("dec 15", "2026-12-15"),
    ("December 15th", "2026-12-15"),
    ("the 15th of december", "2026-12-15"),
    ("12/15", "2026-12-15"),
    ("mid february", "2027-02-15"),
    ("end of january", "2027-01-31"),
    ("early march", "2027-03-01"),
    ("march", "2027-03-01"),
    ("first of january", "2027-01-01"),
    ("jan 5th", "2027-01-05"),
    # extra forms
    ("in a week", "2026-12-08"),
    ("this friday", "2026-12-04"),
    ("January 5th 2027", "2027-01-05"),
    ("2moro", "2026-12-02"),
]


@pytest.mark.parametrize(("span", "expected"), AR)
def test_arabic(span: str, expected: str) -> None:
    assert parse_date(span, "ar", REF) == date.fromisoformat(expected)


@pytest.mark.parametrize(("span", "expected"), FR)
def test_french(span: str, expected: str) -> None:
    assert parse_date(span, "fr", REF) == date.fromisoformat(expected)


@pytest.mark.parametrize(("span", "expected"), EN)
def test_english(span: str, expected: str) -> None:
    assert parse_date(span, "en", REF) == date.fromisoformat(expected)


def test_numeric_order_depends_on_language() -> None:
    assert parse_date("05/01", "fr", REF) == date(2027, 1, 5)
    assert parse_date("05/01", "en", REF) == date(2027, 5, 1)


def test_base_anchors_relative_end_dates() -> None:
    start = date(2027, 3, 3)  # a Wednesday
    assert parse_date("dimanche", "fr", REF, base=start) == date(2027, 3, 7)
    assert parse_date("10 mars", "fr", REF, base=start) == date(2027, 3, 10)


@pytest.mark.parametrize("span", ["", "   ", "bientôt", "n'importe quand", "قريبا"])
def test_unparseable_returns_none(span: str) -> None:
    assert parse_date(span, "fr", REF) is None


DURATIONS = [
    ("ثلاث ليال", 3),
    ("لمدة ثلاث ليال", 3),
    ("ليلتين", 2),
    ("لمدة أسبوع", 7),
    ("أسبوعين", 14),
    ("pour une semaine", 7),
    ("3 nuits", 3),
    ("pour deux nuits", 2),
    ("cinq jours", 5),
    ("for 5 nights", 5),
    ("a week", 7),
    ("two weeks", 14),
    ("a fortnight", 14),
    ("for 10 days", 10),
]


@pytest.mark.parametrize(("span", "days"), DURATIONS)
def test_durations(span: str, days: int) -> None:
    assert parse_duration_days(span) == days


@pytest.mark.parametrize("span", ["10 mars", "dimanche", "demain", "next week", "dans 3 jours"])
def test_dates_are_not_durations(span: str) -> None:
    assert parse_duration_days(span) is None


def test_resolve_range_with_dates() -> None:
    result = resolve_dates("3 mars", "10 mars", "fr", REF)
    assert (result.start, result.end) == (date(2027, 3, 3), date(2027, 3, 10))


def test_resolve_range_across_year_end() -> None:
    result = resolve_dates("28 décembre", "5 janvier", "fr", REF)
    assert (result.start, result.end) == (date(2026, 12, 28), date(2027, 1, 5))


def test_resolve_range_with_duration() -> None:
    result = resolve_dates("الأسبوع القادم", "ثلاث ليال", "ar", REF)
    assert (result.start, result.end) == (date(2026, 12, 8), date(2026, 12, 11))
    result = resolve_dates("ce weekend", "pour une semaine", "fr", REF)
    assert result.end == date(2026, 12, 12)


def test_duration_without_start_gives_no_end() -> None:
    assert resolve_dates(None, "3 nights", "en", REF).end is None


def test_canonical_tokens() -> None:
    assert canonical_tokens("Le Week-End prochain") == ["weekend", "next"]
    assert canonical_tokens("الجمعة القادمة") == ["friday", "next"]
    assert canonical_tokens("15th of December") == ["15", "december"]
