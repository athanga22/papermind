"""Shared fixtures for PaperMind tests."""

from __future__ import annotations

import pytest

from papermind.ingestion.models import Chunk, Section


@pytest.fixture
def sample_section() -> Section:
    return Section(
        name="methodology",
        raw_name="3. Methodology",
        text=(
            "We propose a novel attention mechanism. "
            "The model processes input tokens in parallel using multi-head attention. "
            "Each head attends to different representation subspaces. "
            "The output is concatenated and projected through a linear layer. "
            "\n\n"
            "Training was performed on 8 A100 GPUs for 72 hours. "
            "We used the AdamW optimiser with a learning rate of 3e-4 and weight decay of 0.01. "
            "A cosine learning rate schedule was applied with 1000 warmup steps."
        ),
        page_start=3,
        page_end=5,
    )


@pytest.fixture
def sample_chunk() -> Chunk:
    return Chunk(
        text="We propose a novel attention mechanism for efficient transformer training.",
        paper_title="Attention Is All You Need",
        authors=["Vaswani, A.", "Shazeer, N."],
        section="methodology",
        page_number=3,
        year=2017,
        contains_math=False,
        token_count=12,
        paper_id="abc123",
    )


@pytest.fixture
def sample_chunks(sample_chunk: Chunk) -> list[Chunk]:
    """A small corpus for BM25 / fusion tests."""
    base = sample_chunk
    return [
        base,
        Chunk(
            text="The cross-encoder reranker scores query-document pairs jointly.",
            paper_title="Dense Passage Retrieval",
            authors=["Karpukhin, V."],
            section="methodology",
            page_number=4,
            year=2020,
            contains_math=False,
            token_count=10,
            paper_id="def456",
        ),
        Chunk(
            text="BM25 is a bag-of-words retrieval function that ranks documents based on term frequency.",
            paper_title="Okapi BM25 Overview",
            authors=["Robertson, S."],
            section="introduction",
            page_number=1,
            year=1994,
            contains_math=False,
            token_count=14,
            paper_id="ghi789",
        ),
    ]
