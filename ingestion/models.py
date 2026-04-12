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
