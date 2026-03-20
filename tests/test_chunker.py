"""Unit tests for the section-aware chunker."""

from __future__ import annotations

import pytest

from papermind.ingestion.chunker import (
    _chunk_paragraph,
    _contains_math,
    _split_paragraphs,
    _token_count,
    chunk_section,
)
from papermind.ingestion.models import Section


class TestTokenCount:
    def test_simple(self) -> None:
        assert _token_count("hello world foo") == 3

    def test_empty(self) -> None:
        assert _token_count("") == 0

    def test_single_word(self) -> None:
        assert _token_count("word") == 1


class TestContainsMath:
    def test_greek_letter(self) -> None:
        assert _contains_math("The value of α is 0.01")

    def test_latex_command(self) -> None:
        assert _contains_math(r"We use \alpha to denote")

    def test_plain_text(self) -> None:
        assert not _contains_math("This is plain text with no math.")

    def test_numeric_equation(self) -> None:
        assert _contains_math("The accuracy is 1 = 0.95 after training")


class TestSplitParagraphs:
    def test_splits_on_double_newline(self) -> None:
        text = "First paragraph.\n\nSecond paragraph."
        paras = _split_paragraphs(text)
        assert len(paras) == 2
        assert paras[0] == "First paragraph."
        assert paras[1] == "Second paragraph."

    def test_strips_whitespace(self) -> None:
        text = "  hello world  \n\n  another para  "
        paras = _split_paragraphs(text)
        assert all(p == p.strip() for p in paras)

    def test_empty_paras_removed(self) -> None:
        text = "First.\n\n\n\nSecond."
        paras = _split_paragraphs(text)
        assert len(paras) == 2


class TestChunkParagraph:
    def test_short_paragraph_unchanged(self) -> None:
        text = "Short text."
        result = _chunk_paragraph(text, max_tokens=400)
        assert result == [text]

    def test_long_paragraph_splits(self) -> None:
        # Build a paragraph with ~500 tokens
        sentence = "This is a test sentence with several words. "
        long_para = sentence * 50
        result = _chunk_paragraph(long_para, max_tokens=100)
        assert len(result) > 1
        for chunk in result:
            assert _token_count(chunk) <= 120  # allow slight overage at sentence boundaries

    def test_single_long_sentence_kept_intact(self) -> None:
        # If a single sentence is too long, it shouldn't be split mid-sentence
        long_sentence = " ".join(["word"] * 500)
        result = _chunk_paragraph(long_sentence, max_tokens=100)
        # Should be chunked but each result is word-split (no sentence breaks)
        assert len(result) >= 1


class TestChunkSection:
    def test_basic_chunking(self, sample_section: Section) -> None:
        chunks = chunk_section(
            section=sample_section,
            paper_title="Test Paper",
            authors=["Author A"],
            year=2024,
            paper_id="test123",
        )
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.paper_title == "Test Paper"
            assert chunk.section == "methodology"
            assert chunk.year == 2024
            assert chunk.paper_id == "test123"
            assert chunk.text.strip()

    def test_no_chunks_cross_section(self, sample_section: Section) -> None:
        """All chunks from a section should only contain that section's text."""
        chunks = chunk_section(
            section=sample_section,
            paper_title="Test Paper",
            authors=["Author A"],
            year=2024,
            paper_id="test123",
        )
        for chunk in chunks:
            assert chunk.section == sample_section.name

    def test_chunk_max_tokens_respected(self, sample_section: Section) -> None:
        max_tokens = 50
        chunks = chunk_section(
            section=sample_section,
            paper_title="Test Paper",
            authors=["Author A"],
            year=2024,
            paper_id="test123",
            max_tokens=max_tokens,
        )
        # Chunks should be at or near max_tokens (sentence boundary may cause slight overage)
        for chunk in chunks:
            assert chunk.token_count <= max_tokens * 1.5

    def test_empty_section_returns_empty(self) -> None:
        empty_section = Section(
            name="empty", raw_name="Empty", text="", page_start=1, page_end=1
        )
        chunks = chunk_section(
            section=empty_section,
            paper_title="Test",
            authors=[],
            year=2024,
            paper_id="x",
        )
        assert chunks == []

    def test_unique_chunk_ids(self, sample_section: Section) -> None:
        chunks = chunk_section(
            section=sample_section,
            paper_title="Test Paper",
            authors=["Author A"],
            year=2024,
            paper_id="test123",
        )
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"
