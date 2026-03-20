"""
Section-aware chunking.

Rules (from the plan):
  1. Respect paragraph boundaries by default.
  2. If a paragraph exceeds MAX_TOKENS → split at sentence boundaries.
  3. If a paragraph is below MIN_TOKENS → merge with next paragraph in same section.
  4. Never split across section boundaries.

Token counting uses a simple whitespace tokeniser (fast, no model dependency).
For production you'd swap this for the model's actual tokeniser, but the
~20% error margin is acceptable for chunking purposes.
"""

from __future__ import annotations

import logging
import re
import unicodedata

from ..config import settings
from .models import Chunk, Paper, Section

logger = logging.getLogger(__name__)

# Regex patterns
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_MATH_INDICATORS = re.compile(
    r"[\u0391-\u03C9\u2200-\u22FF]"  # Greek letters + math symbols
    r"|\\[a-zA-Z]+"                   # LaTeX commands
    r"|\$.*?\$"                       # Inline LaTeX
    r"|\d+\s*[=<>≤≥±×÷]\s*\d+",     # Numeric equations
    re.UNICODE,
)


def _token_count(text: str) -> int:
    """Approximate token count via whitespace split (good enough for chunking)."""
    return len(text.split())


def _contains_math(text: str) -> bool:
    return bool(_MATH_INDICATORS.search(text))


def _clean_text(text: str) -> str:
    """Normalise whitespace and fix common PDF extraction artefacts."""
    # Fix hyphenated line breaks common in academic PDFs: "meth-\nod" → "method"
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text)
    # Normalise unicode (NFC)
    text = unicodedata.normalize("NFC", text)
    return text.strip()


def _split_paragraphs(text: str) -> list[str]:
    """Split text on blank lines (paragraph separator in most academic PDFs)."""
    paras = re.split(r"\n\s*\n", text)
    return [_clean_text(p) for p in paras if _clean_text(p)]


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def _chunk_paragraph(
    para: str,
    max_tokens: int,
) -> list[str]:
    """
    If a paragraph exceeds max_tokens, split it at sentence boundaries.
    Returns a list of sub-chunks, each <= max_tokens where possible.
    """
    if _token_count(para) <= max_tokens:
        return [para]

    sentences = _split_sentences(para)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = _token_count(sent)
        if current_tokens + sent_tokens > max_tokens and current:
            chunks.append(" ".join(current))
            current = [sent]
            current_tokens = sent_tokens
        else:
            current.append(sent)
            current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_section(
    section: Section,
    paper_title: str,
    authors: list[str],
    year: int,
    paper_id: str,
    max_tokens: int = settings.chunk_max_tokens,
    min_tokens: int = settings.chunk_min_tokens,
) -> list[Chunk]:
    """
    Convert a Section into Chunks, applying all chunking rules.

    Never produces chunks that cross section boundaries.
    """
    paragraphs = _split_paragraphs(section.text)
    if not paragraphs:
        return []

    # Step 1: split any over-long paragraph at sentence boundaries
    raw_chunks: list[str] = []
    for para in paragraphs:
        raw_chunks.extend(_chunk_paragraph(para, max_tokens))

    # Step 2: merge under-sized chunks with the next one (within section)
    merged: list[str] = []
    i = 0
    while i < len(raw_chunks):
        chunk_text = raw_chunks[i]
        while (
            _token_count(chunk_text) < min_tokens
            and i + 1 < len(raw_chunks)
        ):
            i += 1
            chunk_text = chunk_text + " " + raw_chunks[i]
        merged.append(chunk_text)
        i += 1

    # Step 3: build Chunk objects with metadata
    chunks: list[Chunk] = []
    for text in merged:
        if not text.strip():
            continue
        chunks.append(
            Chunk(
                text=text,
                paper_title=paper_title,
                authors=authors,
                section=section.name,
                page_number=section.page_start,
                year=year,
                contains_math=_contains_math(text),
                token_count=_token_count(text),
                paper_id=paper_id,
            )
        )

    return chunks


def chunk_paper(paper: Paper) -> list[Chunk]:
    """
    Chunk all sections of a Paper, respecting section boundaries.

    Assigns chunks back to paper.chunks and returns them.
    """
    all_chunks: list[Chunk] = []

    for section in paper.sections:
        section_chunks = chunk_section(
            section=section,
            paper_title=paper.title,
            authors=paper.authors,
            year=paper.year,
            paper_id=paper.paper_id,
        )
        all_chunks.extend(section_chunks)

    paper.chunks = all_chunks
    logger.info(
        "Chunked '%s': %d sections → %d chunks", paper.title, len(paper.sections), len(all_chunks)
    )
    return all_chunks
