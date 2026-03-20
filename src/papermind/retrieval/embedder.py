"""Dense embedding wrapper around sentence-transformers."""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import settings

logger = logging.getLogger(__name__)


class Embedder:
    """Thin wrapper that normalises outputs and supports batch encoding."""

    def __init__(self, model_name: str = settings.embedding_model) -> None:
        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        self.dim: int = self._model.get_sentence_embedding_dimension() or settings.embedding_dim

    def embed(self, text: str) -> list[float]:
        """Embed a single string. Returns a normalised float list."""
        vec: np.ndarray = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Embed a list of strings efficiently."""
        vecs: np.ndarray = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
        )
        return vecs.tolist()


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Module-level singleton — model loads once per process."""
    return Embedder()
