"""PDF parsing with PyMuPDF — preserves academic paper section structure."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import fitz  # PyMuPDF

from .models import Paper, Section

logger = logging.getLogger(__name__)

# Patterns that match academic section headings.
# Order matters: more specific patterns first.
_SECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^abstract$", re.I),
    re.compile(r"^\d+\.?\s+introduction\b", re.I),
    re.compile(r"^introduction$", re.I),
    re.compile(r"^\d+\.?\s+related\s+work\b", re.I),
    re.compile(r"^\d+\.?\s+background\b", re.I),
    re.compile(r"^\d+\.?\s+(method|methodology|approach|model)\b", re.I),
    re.compile(r"^\d+\.?\s+(experiment|experimental\s+setup)\b", re.I),
    re.compile(r"^\d+\.?\s+result", re.I),
    re.compile(r"^\d+\.?\s+discussion\b", re.I),
    re.compile(r"^\d+\.?\s+conclusion", re.I),
    re.compile(r"^\d+\.?\s+reference", re.I),
    re.compile(r"^reference", re.I),
    re.compile(r"^\d+\.?\s+appendix\b", re.I),
]

# Heuristic: bold or large font usually means a heading
_MIN_HEADING_FONT_SIZE = 10.5
_HEADING_FONT_WEIGHT = "bold"  # flag check via flags & 2**4


def _is_heading(span: dict) -> bool:  # type: ignore[type-arg]
    """Return True if a text span looks like a section heading."""
    text = span["text"].strip()
    if not text or len(text) > 120:
        return False
    is_bold = bool(span["flags"] & 2**4)
    is_large = span["size"] >= _MIN_HEADING_FONT_SIZE
    return (is_bold or is_large) and any(p.match(text) for p in _SECTION_PATTERNS)


def _normalise_section_name(raw: str) -> str:
    """Strip leading numbering and lowercase, e.g. '3. Methodology' -> 'methodology'."""
    cleaned = re.sub(r"^\d+\.?\s*", "", raw).strip().lower()
    return cleaned or raw.lower()


def _extract_year(doc: fitz.Document) -> int:
    """Best-effort year extraction from PDF metadata or first-page text."""
    meta = doc.metadata or {}
    for field in ("creationDate", "modDate"):
        val = meta.get(field, "")
        m = re.search(r"(20\d{2})", val)
        if m:
            return int(m.group(1))

    # Fall back: scan the first two pages for a 4-digit year
    for page_idx in range(min(2, len(doc))):
        text = doc[page_idx].get_text()
        m = re.search(r"\b(20\d{2})\b", text)
        if m:
            return int(m.group(1))

    return 0


def _extract_authors(doc: fitz.Document) -> list[str]:
    """Extract author names from PDF metadata or first-page heuristics."""
    meta = doc.metadata or {}
    author_str = meta.get("author", "").strip()
    if author_str:
        # Split on common separators
        authors = re.split(r"[,;]|\band\b", author_str, flags=re.I)
        return [a.strip() for a in authors if a.strip()]

    # Heuristic: look for author-like lines on the first page
    # (short lines after the title, before the abstract)
    first_page_text = doc[0].get_text() if len(doc) > 0 else ""
    lines = [ln.strip() for ln in first_page_text.split("\n") if ln.strip()]
    candidates: list[str] = []
    for ln in lines[1:8]:  # skip title line, check next few
        if re.match(r"^[A-Z][a-z]+(\s[A-Z][a-z]+)+$", ln):
            candidates.append(ln)
    return candidates or ["Unknown"]


def parse_pdf(path: str | Path) -> Paper:
    """
    Parse a PDF into a Paper with structured Sections.

    Extracts:
    - Title (first non-empty line or metadata)
    - Authors (metadata or heuristic)
    - Publication year (metadata or heuristic)
    - Sections with their text and page range
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    logger.info("Parsing %s", path.name)
    doc = fitz.open(str(path))

    title = (doc.metadata or {}).get("title", "").strip() or path.stem
    authors = _extract_authors(doc)
    year = _extract_year(doc)

    # ── Collect all text blocks with their font metadata ─────────────────
    blocks: list[dict] = []  # type: ignore[type-arg]
    for page_idx, page in enumerate(doc):
        page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:  # 0 = text block
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    blocks.append(
                        {
                            "text": span["text"],
                            "size": span["size"],
                            "flags": span["flags"],
                            "page": page_idx + 1,  # 1-indexed
                        }
                    )

    # ── Walk blocks and split into sections ──────────────────────────────
    sections: list[Section] = []
    current_heading: str | None = None
    current_page_start: int = 1
    buffer: list[str] = []
    buffer_pages: list[int] = []

    def _flush_section(next_page: int) -> None:
        nonlocal current_heading, buffer, buffer_pages, current_page_start
        if current_heading is None:
            return
        text = " ".join(buffer).strip()
        if text:
            page_end = buffer_pages[-1] if buffer_pages else next_page
            sections.append(
                Section(
                    name=_normalise_section_name(current_heading),
                    raw_name=current_heading,
                    text=text,
                    page_start=current_page_start,
                    page_end=page_end,
                )
            )
        buffer = []
        buffer_pages = []

    for blk in blocks:
        text = blk["text"].strip()
        if not text:
            continue

        if _is_heading(blk):
            _flush_section(blk["page"])
            current_heading = text
            current_page_start = blk["page"]
        else:
            if current_heading is None:
                # Text before the first heading — treat as a preamble section
                current_heading = "preamble"
                current_page_start = blk["page"]
            buffer.append(text)
            buffer_pages.append(blk["page"])

    _flush_section(next_page=len(doc))
    doc.close()

    # Deduplicate / merge tiny adjacent same-named sections
    merged = _merge_duplicate_sections(sections)

    logger.info(
        "Parsed '%s': %d sections, %d pages", title, len(merged), len(doc)  # type: ignore[arg-type]
    )
    return Paper(title=title, authors=authors, year=year, path=str(path), sections=merged)


def _merge_duplicate_sections(sections: list[Section]) -> list[Section]:
    """Merge consecutive sections that have the same normalised name."""
    if not sections:
        return sections
    merged: list[Section] = [sections[0]]
    for sec in sections[1:]:
        prev = merged[-1]
        if sec.name == prev.name:
            merged[-1] = Section(
                name=prev.name,
                raw_name=prev.raw_name,
                text=prev.text + " " + sec.text,
                page_start=prev.page_start,
                page_end=sec.page_end,
            )
        else:
            merged.append(sec)
    return merged
