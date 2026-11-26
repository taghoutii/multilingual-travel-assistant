from __future__ import annotations

import math

import pytest

from travel_agent.eval.gate_metrics import Rate, grouped, percentile, rate


def test_wilson_interval_known_value() -> None:
    lo, hi = Rate(8, 10).wilson()
    assert (lo, hi) == (pytest.approx(0.4902, abs=1e-4), pytest.approx(0.9433, abs=1e-4))


def test_wilson_bounds_at_extremes() -> None:
    lo, hi = Rate(10, 10).wilson()
    assert hi == pytest.approx(1.0) and 0.6 < lo < 0.75
    lo, hi = Rate(0, 10).wilson()
    assert lo == 0.0 and 0.25 < hi < 0.4


def test_empty_rate() -> None:
    assert math.isnan(Rate(0, 0).value)
    assert Rate(0, 0).fmt() == "n/a"


def test_rate_and_format() -> None:
    r = rate([True, True, False, True])
    assert (r.hits, r.n, r.value) == (3, 4, 0.75)
    assert r.fmt().startswith("75.0% [") and r.fmt().endswith("(3/4)")


def test_grouped() -> None:
    groups = grouped([("fr", True), ("en", False), ("fr", False), ("ar", True)])
    assert list(groups) == ["ar", "en", "fr"]
    assert (groups["fr"].hits, groups["fr"].n) == (1, 2)


def test_percentile() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 50) == 2.5
    assert percentile(values, 0) == 1.0
    assert percentile(values, 100) == 4.0
    assert math.isnan(percentile([], 50))
