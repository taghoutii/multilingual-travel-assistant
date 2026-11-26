"""Sentence embeddings (needs the `ml` dependency group).

`Embedder` is the interface the scope gate and the leakage check depend on, so tests can pass a
small fake instead of loading sentence-transformers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from travel_agent.config import PROJECT_ROOT, get_settings

HF_CACHE = PROJECT_ROOT / "models" / "hf"

# texts -> array of shape (len(texts), dim), rows L2-normalized
Embedder = Callable[[list[str]], Any]


def load_embedder(model_name: str | None = None, batch_size: int = 64) -> Embedder:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        model_name or get_settings().embeddings_model, cache_folder=str(HF_CACHE)
    )

    def embed(texts: list[str]) -> Any:
        return model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
        )

    return embed
