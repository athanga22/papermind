"""Domain models for the ingestion pipeline."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any


# Canonical academic section ordering — used for sorting and boundary detection
SECTION_ORDER = [
    "abstract",
    "introduction",
    "related work",
    "background",
    "methodology",
    "method",
    "approach",
    "model",
    "experiments",
    "experimental setup",
    "results",
    "discussion",
    "conclusion",
    "references",
    "appendix",
]


@dataclass
class Section:
    name: str          # Normalised section name, e.g. "methodology"
    raw_name: str      # Heading text as it appears in the PDF
    text: str          # Raw section text (pre-chunking)
    page_start: int
    page_end: int

    @property
    def order_index(self) -> int:
        """Return canonical section order index, or 999 if unknown."""
        name_lower = self.name.lower()
        for i, canonical in enumerate(SECTION_ORDER):
            if canonical in name_lower:
                return i
        return 999


@dataclass
class Chunk:
    """A single retrievable unit — one entry in Qdrant."""

    text: str
    paper_title: str
    authors: list[str]
    section: str          # Normalised section name
    page_number: int      # Page where the chunk starts
    year: int
    contains_math: bool   # True if math symbols detected in text
    token_count: int

    # Stable IDs
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    paper_id: str = ""    # Set by caller from paper path hash

    def to_payload(self) -> dict[str, Any]:
        """Serialise to Qdrant point payload (all scalar values)."""
        return {
            "text": self.text,
            "paper_title": self.paper_title,
            "authors": self.authors,
            "section": self.section,
            "page_number": self.page_number,
            "year": self.year,
            "contains_math": self.contains_math,
            "token_count": self.token_count,
            "paper_id": self.paper_id,
        }

    @classmethod
    def from_payload(cls, chunk_id: str, payload: dict[str, Any]) -> "Chunk":
        return cls(
            text=payload["text"],
            paper_title=payload["paper_title"],
            authors=payload["authors"],
            section=payload["section"],
            page_number=payload["page_number"],
            year=payload["year"],
            contains_math=payload["contains_math"],
            token_count=payload["token_count"],
            chunk_id=chunk_id,
            paper_id=payload.get("paper_id", ""),
        )


@dataclass
class Paper:
    title: str
    authors: list[str]
    year: int
    path: str
    sections: list[Section] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)

    @property
    def paper_id(self) -> str:
        """Stable ID derived from the file path."""
        return hashlib.sha256(self.path.encode()).hexdigest()[:16]


@dataclass
class RetrievedChunk:
    """A Chunk returned from retrieval, augmented with its score."""

    chunk: Chunk
    score: float
    rank: int  # Final rank after fusion/reranking

    @property
    def citation(self) -> str:
        return f"[{self.chunk.paper_title}, {self.chunk.section.title()}, p.{self.chunk.page_number}]"
