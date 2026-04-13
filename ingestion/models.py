"""
Core data models for the ingestion pipeline.
Models grow richer as steps are added — only what each step needs is defined here.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParsedPaper:
    """Output of Step 1 (LlamaParse). Raw markdown from one PDF."""
    paper_id: str        # 12-char md5 hex of the filename
    file_name: str
    file_path: str
    markdown: str        # Full document markdown, all pages joined


@dataclass
class Chunk:
    """Output of Steps 2-3. One indexable unit of a paper."""
    chunk_id: str
    paper_id: str
    paper_title: str
    authors: list[str]
    year: Optional[int]
    section: str         # e.g. "Introduction", "Methods"
    chunk_index: int     # position within the paper
    text: str
    is_table: bool = False
    contains_math: bool = False
    embedding: Optional[list[float]] = field(default=None, repr=False)


@dataclass
class Bibliography:
    """Extracted bibliography from one paper (Step 3)."""
    paper_id: str
    references: list[str]   # raw reference strings


def contextualize_chunk(chunk: Chunk) -> str:
    """
    Prepend paper-level context to a chunk's text for embedding and BM25 indexing.

    Why: Chunks like "The results show 15% improvement" are ambiguous without
    knowing which paper and section they belong to. Prepending this context
    makes embeddings more discriminative and lets BM25 match on paper titles
    and section names.

    The raw chunk.text stays in the Qdrant payload for display/synthesis —
    this function is only used at embed/index time.

    See: Anthropic's "Contextual Retrieval" (2024) — metadata prefix alone
    captures most of the benefit without requiring an LLM call per chunk.
    """
    parts = []
    if chunk.paper_title:
        parts.append(f"Paper: {chunk.paper_title}")
    if chunk.authors:
        parts.append(f"Authors: {', '.join(chunk.authors[:3])}")
    if chunk.year:
        parts.append(f"Year: {chunk.year}")
    if chunk.section:
        parts.append(f"Section: {chunk.section}")

    header = ". ".join(parts) + ".\n\n" if parts else ""
    return header + chunk.text
