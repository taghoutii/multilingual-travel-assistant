from __future__ import annotations

import random

from travel_agent.seed import set_seed


def test_set_seed_is_reproducible() -> None:
    set_seed(123)
    first = [random.random() for _ in range(3)]
    set_seed(123)
    assert [random.random() for _ in range(3)] == first
