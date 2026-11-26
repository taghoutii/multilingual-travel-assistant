"""Scope gate on small synthetic embeddings (no embedding model)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from travel_agent.scope_gate import ScopeGate, fit_svm, threshold_for_accept, tune


def unit_rows(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def cluster(rng: np.random.Generator, centre: np.ndarray, n: int, spread: float) -> np.ndarray:
    return unit_rows(centre + spread * rng.standard_normal((n, len(centre))))


@pytest.fixture
def data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    dim = 16
    inside, outside = np.zeros(dim), np.zeros(dim)
    inside[0], outside[1] = 1.0, 1.0
    return {
        "train": cluster(rng, inside, 200, 0.15),
        "val_in": cluster(rng, inside, 100, 0.15),
        "val_off": cluster(rng, outside, 100, 0.15),
    }


def test_threshold_accepts_at_least_target() -> None:
    scores = np.arange(100, dtype=float)
    t = threshold_for_accept(scores, 0.97)
    assert t == 3.0
    assert np.mean(scores >= t) == pytest.approx(0.97)
    assert threshold_for_accept(scores, 1.0) == 0.0
    assert np.mean(scores >= threshold_for_accept(np.arange(7.0), 0.9)) >= 0.9


def test_threshold_needs_scores() -> None:
    with pytest.raises(ValueError):
        threshold_for_accept(np.array([]), 0.97)


def test_tune_meets_target_and_separates_clusters(data: dict[str, np.ndarray]) -> None:
    result = tune(data["train"], data["val_in"], data["val_off"], [0.01, 0.1], ["scale", 2.0], 0.97)
    assert len(result.grid) == 4
    chosen = next(r for r in result.grid if (r.nu, r.gamma) == (result.nu, result.gamma))
    assert chosen.in_scope_accept >= 0.97
    assert chosen.off_topic_reject == max(r.off_topic_reject for r in result.grid)
    assert chosen.off_topic_reject > 0.95


def test_tie_prefers_smaller_nu(data: dict[str, np.ndarray]) -> None:
    result = tune(data["train"], data["val_in"], data["val_off"], [0.1, 0.01], [2.0], 0.97)
    if result.grid[0].off_topic_reject == result.grid[1].off_topic_reject:
        assert result.nu == 0.01


def test_decide_save_and_load(tmp_path: Path, data: dict[str, np.ndarray]) -> None:
    vectors = {"inside": data["val_in"][0], "outside": data["val_off"][0]}

    def embed(texts: list[str]) -> np.ndarray:
        return np.stack([vectors[t] for t in texts])

    svm = fit_svm(data["train"], 0.05, 2.0)
    threshold = threshold_for_accept(svm.decision_function(data["val_in"]), 0.97)
    gate = ScopeGate(svm, threshold, embed, meta={"embeddings_model": "fake"})
    assert gate.decide("inside").in_scope
    assert not gate.decide("outside").in_scope

    path = tmp_path / "gate.joblib"
    gate.save(path)
    loaded = ScopeGate.load(path, embed)
    assert loaded.threshold == threshold
    assert loaded.meta == {"embeddings_model": "fake"}
    assert loaded.decide("outside").score == pytest.approx(gate.decide("outside").score)


def test_scores_need_an_embedder(data: dict[str, np.ndarray]) -> None:
    gate = ScopeGate(fit_svm(data["train"], 0.05, 2.0), 0.0)
    with pytest.raises(RuntimeError):
        gate.decide("anything")
