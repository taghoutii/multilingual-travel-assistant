"""Small metrics for the language ID and scope gate evaluation: rates with Wilson 95% intervals,
grouped rates and latency percentiles."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

Z95 = 1.959963984540054


@dataclass(frozen=True)
class Rate:
    hits: int
    n: int

    @property
    def value(self) -> float:
        return self.hits / self.n if self.n else float("nan")

    def wilson(self, z: float = Z95) -> tuple[float, float]:
        """Wilson score interval (95% by default); (nan, nan) when n == 0."""
        if self.n == 0:
            return (float("nan"), float("nan"))
        p, n = self.value, self.n
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return (max(0.0, centre - half), min(1.0, centre + half))

    def fmt(self) -> str:
        if self.n == 0:
            return "n/a"
        lo, hi = self.wilson()
        return f"{100 * self.value:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}] ({self.hits}/{self.n})"


def rate(flags: Iterable[bool]) -> Rate:
    values = list(flags)
    return Rate(sum(values), len(values))


def grouped(items: Iterable[tuple[str, bool]]) -> dict[str, Rate]:
    """{group: Rate} from (group, success) pairs, groups sorted by name."""
    buckets: dict[str, list[bool]] = {}
    for group, ok in items:
        buckets.setdefault(group, []).append(ok)
    return {g: rate(buckets[g]) for g in sorted(buckets)}


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile, q in [0, 100]."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
