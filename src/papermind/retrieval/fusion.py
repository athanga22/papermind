"""
Reciprocal Rank Fusion (RRF).

RRF is used to merge ranked lists from dense and sparse retrieval without
needing to normalise incompatible score scales.

Formula: RRF(d) = Σ_r 1 / (k + rank_r(d))
where k is a smoothing constant (default 60, from the original paper).
"""

from __future__ import annotations

from ..config import settings
from ..ingestion.models import Chunk, RetrievedChunk


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[Chunk, float]]],
    top_k: int = settings.rerank_top_k,
    k: int = settings.rrf_k,
) -> list[RetrievedChunk]:
    """
    Fuse multiple ranked lists via RRF.

    Args:
        ranked_lists: Each inner list is [(chunk, score), ...] sorted desc.
                      Scores are ignored — only ranks matter.
        top_k:        Number of results to return.
        k:            RRF smoothing constant. Higher = more weight to lower-ranked docs.

    Returns:
        List of RetrievedChunk sorted by fused score descending.
    """
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, Chunk] = {}

    for ranked in ranked_lists:
        for rank, (chunk, _) in enumerate(ranked, start=1):
            cid = chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
            chunk_map[cid] = chunk

    # Sort by fused score descending
    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

    return [
        RetrievedChunk(
            chunk=chunk_map[cid],
            score=rrf_scores[cid],
            rank=i + 1,
        )
        for i, cid in enumerate(sorted_ids[:top_k])
    ]
