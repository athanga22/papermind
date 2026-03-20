"""
Cross-encoder reranker.

Takes the top-N candidates from RRF fusion and produces a final ranking
with calibrated relevance scores. The top-score is also used as a
confidence gate: if it falls below RERANKER_THRESHOLD, the agent loop
triggers query rewriting.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import CrossEncoder

from ..config import settings
from ..ingestion.models import RetrievedChunk

logger = logging.getLogger(__name__)


class Reranker:
    """Thin wrapper around a cross-encoder model."""

    def __init__(self, model_name: str = settings.reranker_model) -> None:
        logger.info("Loading reranker model: %s", model_name)
        self._model: CrossEncoder = CrossEncoder(model_name)  # type: ignore[assignment]

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int = settings.rerank_top_k,
    ) -> list[RetrievedChunk]:
        """
        Rerank candidates using the cross-encoder.

        Returns top_k RetrievedChunks with updated scores and ranks.
        Scores are sigmoid-normalised to [0, 1].
        """
        if not candidates:
            return []

        pairs = [(query, c.chunk.text) for c in candidates]
        raw_scores: np.ndarray = self._model.predict(pairs)

        # Sigmoid to [0, 1] — cross-encoders output logits by default
        scores: list[float] = (1.0 / (1.0 + np.exp(-raw_scores))).tolist()

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            RetrievedChunk(chunk=rc.chunk, score=score, rank=i + 1)
            for i, (rc, score) in enumerate(ranked[:top_k])
        ]

    def top_score(self, reranked: list[RetrievedChunk]) -> float:
        """Return the score of the top-ranked chunk, or 0.0 if empty."""
        return reranked[0].score if reranked else 0.0

    def is_confident(
        self,
        reranked: list[RetrievedChunk],
        threshold: float = settings.reranker_threshold,
    ) -> bool:
        """Return True if the best chunk's score meets the confidence threshold."""
        return self.top_score(reranked) >= threshold


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    """Module-level singleton."""
    return Reranker()
