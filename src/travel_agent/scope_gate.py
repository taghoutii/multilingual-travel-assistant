"""Scope gate (pipeline stage 2): is the message a request the 15 subcategories cover?

Sentence embeddings (L2-normalized) -> One-Class SVM fitted on in-scope training queries only.
The SVM's decision function is a score (higher = more like the training queries); a message is in
scope when its score is at or above a threshold chosen on validation so that a target share of
in-scope validation queries is accepted (default 97%). `nu` and `gamma` are chosen on validation
too: the pair with the highest off-topic rejection at that threshold.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.svm import OneClassSVM

from travel_agent.config import PROJECT_ROOT, get_settings
from travel_agent.embeddings import Embedder

MODELS_DIR = PROJECT_ROOT / "models"
Gamma = float | str


@dataclass(frozen=True)
class ScopeDecision:
    in_scope: bool
    score: float
    threshold: float


@dataclass(frozen=True)
class GridRow:
    nu: float
    gamma: Gamma
    threshold: float
    in_scope_accept: float
    off_topic_reject: float


@dataclass(frozen=True)
class TuneResult:
    nu: float
    gamma: Gamma
    threshold: float
    svm: OneClassSVM
    grid: list[GridRow] = field(default_factory=list)


def threshold_for_accept(scores: np.ndarray, target: float) -> float:
    """Highest threshold that accepts at least `target` of `scores` (accept = score >= t)."""
    if len(scores) == 0:
        raise ValueError("no scores")
    ordered = np.sort(scores)
    n_reject = math.floor((1.0 - target) * len(ordered) + 1e-9)
    return float(ordered[n_reject])


def fit_svm(x_train: np.ndarray, nu: float, gamma: Gamma) -> OneClassSVM:
    return OneClassSVM(kernel="rbf", nu=nu, gamma=gamma).fit(x_train)


def tune(
    x_train: np.ndarray,
    x_val_in: np.ndarray,
    x_val_off: np.ndarray,
    nu_grid: Sequence[float],
    gamma_grid: Sequence[Gamma],
    target: float,
) -> TuneResult:
    """Grid over (nu, gamma); keep the best off-topic rejection, ties -> smaller nu."""
    best: TuneResult | None = None
    best_key: tuple[float, float, float] | None = None
    grid: list[GridRow] = []
    for nu in nu_grid:
        for gamma in gamma_grid:
            svm = fit_svm(x_train, nu, gamma)
            s_in = svm.decision_function(x_val_in)
            s_off = svm.decision_function(x_val_off)
            t = threshold_for_accept(s_in, target)
            row = GridRow(nu, gamma, t, float(np.mean(s_in >= t)), float(np.mean(s_off < t)))
            grid.append(row)
            key = (row.off_topic_reject, row.in_scope_accept, -nu)
            if best_key is None or key > best_key:
                best_key, best = key, TuneResult(nu, gamma, t, svm)
    assert best is not None
    return TuneResult(best.nu, best.gamma, best.threshold, best.svm, grid)


class ScopeGate:
    def __init__(
        self,
        svm: OneClassSVM,
        threshold: float,
        embedder: Embedder | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.svm = svm
        self.threshold = threshold
        self.embedder = embedder
        self.meta = meta or {}

    def scores(self, texts: list[str]) -> np.ndarray:
        if self.embedder is None:
            raise RuntimeError("ScopeGate has no embedder")
        return np.asarray(self.svm.decision_function(np.asarray(self.embedder(texts))))

    def decide(self, text: str) -> ScopeDecision:
        score = float(self.scores([text])[0])
        return ScopeDecision(score >= self.threshold, score, self.threshold)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"svm": self.svm, "threshold": self.threshold, "meta": self.meta}, path)

    @classmethod
    def load(cls, path: Path, embedder: Embedder | None = None) -> ScopeGate:
        blob = joblib.load(path)
        return cls(blob["svm"], float(blob["threshold"]), embedder, blob.get("meta", {}))


def model_path() -> Path:
    return MODELS_DIR / get_settings().scope_gate.model_file


@lru_cache(maxsize=1)
def get_scope_gate() -> ScopeGate:
    from travel_agent.embeddings import load_embedder

    gate = ScopeGate.load(model_path(), load_embedder())
    trained_with = gate.meta.get("embeddings_model")
    if trained_with != get_settings().embeddings_model:
        raise RuntimeError(f"scope gate trained with {trained_with!r}; run train_scope_gate.py")
    return gate
