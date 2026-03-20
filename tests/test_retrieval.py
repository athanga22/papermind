"""Unit tests for retrieval components (BM25, RRF — no network required)."""

from __future__ import annotations

import pytest

from papermind.ingestion.models import Chunk, RetrievedChunk
from papermind.retrieval.fusion import reciprocal_rank_fusion
from papermind.retrieval.sparse import BM25Index


class TestBM25Index:
    def test_build_and_search(self, sample_chunks: list[Chunk]) -> None:
        idx = BM25Index()
        idx.build(sample_chunks)
        results = idx.search("attention mechanism transformer", top_k=3)
        assert len(results) > 0
        assert all(isinstance(c, Chunk) for c, _ in results)
        assert all(isinstance(s, float) for _, s in results)

    def test_results_sorted_descending(self, sample_chunks: list[Chunk]) -> None:
        idx = BM25Index()
        idx.build(sample_chunks)
        results = idx.search("retrieval ranking", top_k=3)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self, sample_chunks: list[Chunk]) -> None:
        idx = BM25Index()
        idx.build(sample_chunks)
        results = idx.search("attention", top_k=2)
        assert len(results) <= 2

    def test_raises_without_build(self) -> None:
        idx = BM25Index()
        with pytest.raises(RuntimeError, match="not built"):
            idx.search("test query")

    def test_relevant_doc_ranked_first(self, sample_chunks: list[Chunk]) -> None:
        idx = BM25Index()
        idx.build(sample_chunks)
        # The BM25 chunk is about BM25 — should rank top for this query
        results = idx.search("BM25 term frequency retrieval function", top_k=3)
        top_chunk = results[0][0]
        assert "BM25" in top_chunk.text or "bm25" in top_chunk.text.lower()


class TestRRF:
    def _make_retrieved(self, chunk: Chunk, score: float = 1.0) -> RetrievedChunk:
        return RetrievedChunk(chunk=chunk, score=score, rank=1)

    def test_basic_fusion(self, sample_chunks: list[Chunk]) -> None:
        list1 = [(sample_chunks[0], 0.9), (sample_chunks[1], 0.7), (sample_chunks[2], 0.5)]
        list2 = [(sample_chunks[2], 0.8), (sample_chunks[0], 0.6), (sample_chunks[1], 0.4)]
        result = reciprocal_rank_fusion([list1, list2], top_k=3)
        assert len(result) == 3
        assert all(isinstance(r, RetrievedChunk) for r in result)

    def test_ranks_are_sequential(self, sample_chunks: list[Chunk]) -> None:
        list1 = [(c, 0.9 - i * 0.1) for i, c in enumerate(sample_chunks)]
        result = reciprocal_rank_fusion([list1], top_k=3)
        ranks = [r.rank for r in result]
        assert ranks == list(range(1, len(result) + 1))

    def test_scores_descending(self, sample_chunks: list[Chunk]) -> None:
        list1 = [(c, 0.9 - i * 0.1) for i, c in enumerate(sample_chunks)]
        result = reciprocal_rank_fusion([list1], top_k=3)
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self, sample_chunks: list[Chunk]) -> None:
        list1 = [(c, 1.0) for c in sample_chunks]
        result = reciprocal_rank_fusion([list1], top_k=2)
        assert len(result) == 2

    def test_deduplication_across_lists(self, sample_chunks: list[Chunk]) -> None:
        """A chunk appearing in both lists should appear once in output."""
        shared = sample_chunks[0]
        list1 = [(shared, 0.9), (sample_chunks[1], 0.7)]
        list2 = [(shared, 0.8), (sample_chunks[2], 0.6)]
        result = reciprocal_rank_fusion([list1, list2], top_k=10)
        ids = [r.chunk.chunk_id for r in result]
        assert len(ids) == len(set(ids)), "Duplicates in RRF output"

    def test_chunk_appearing_in_both_lists_boosted(self, sample_chunks: list[Chunk]) -> None:
        """A chunk in both lists should outscore a chunk in only one list at same rank."""
        shared = sample_chunks[0]
        unique = sample_chunks[1]
        list1 = [(shared, 0.9), (unique, 0.8)]
        list2 = [(shared, 0.9)]  # shared appears in both
        result = reciprocal_rank_fusion([list1, list2], top_k=3)
        result_map = {r.chunk.chunk_id: r.score for r in result}
        assert result_map[shared.chunk_id] > result_map[unique.chunk_id]

    def test_empty_lists(self) -> None:
        result = reciprocal_rank_fusion([], top_k=5)
        assert result == []

    def test_single_empty_list(self) -> None:
        result = reciprocal_rank_fusion([[]], top_k=5)
        assert result == []
