"""
BM25 sparse index using rank_bm25.

Stored in memory and rebuilt on ingest. For a production system you'd
persist this to disk alongside Qdrant; for a portfolio project, rebuilding
on startup from Qdrant payloads is acceptable and keeps the architecture simple.
"""

from __future__ import annotations

import logging
import re
import string

from rank_bm25 import BM25Okapi

from ..ingestion.models import Chunk

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "this", "that", "these",
        "those", "it", "its",
    }
)


def _tokenise(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stopwords."""
    text = text.lower()
    text = re.sub(r"[" + re.escape(string.punctuation) + r"]", " ", text)
    return [tok for tok in text.split() if tok and tok not in _STOPWORDS]


class BM25Index:
    """
    In-memory BM25 index over a list of Chunks.

    The index maps directly to chunk positions in self.chunks,
    so callers can zip scores with chunks.
    """

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def build(self, chunks: list[Chunk]) -> None:
        """Build (or rebuild) the index from a list of Chunks."""
        if not chunks:
            logger.warning("BM25Index.build called with empty chunk list")
            return
        self.chunks = chunks
        tokenised = [_tokenise(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenised)
        logger.info("BM25 index built: %d documents", len(chunks))

    def search(self, query: str, top_k: int = 20) -> list[tuple[Chunk, float]]:
        """
        Return top-k (chunk, score) pairs for the query.

        Scores are BM25 raw scores — not normalised. RRF fusion handles
        the incompatibility with dense scores downstream.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built. Call build() first.")

        tokens = _tokenise(query)
        scores: list[float] = self._bm25.get_scores(tokens).tolist()

        # Pair and sort descending
        ranked = sorted(
            zip(self.chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]
