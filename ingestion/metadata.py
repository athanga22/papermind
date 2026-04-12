"""
Step 3 — Metadata extraction from parsed markdown.

Extracts per-paper metadata (title, authors, year) and bibliography.
All extraction is regex/heuristic — no LLM calls.

arXiv paper format (standard):
  Line 1:  arXiv:{id}v{N} [{category}] {day} {month} {year}
  # Title
  Author1   Author2   Author3    (space/tab separated, or comma separated)
  affiliations, emails...
  # Abstract
  ...
  # References
  [1] ...
"""

import re
from typing import Optional

from ingestion.models import Bibliography, ParsedPaper

# ── Institution / email line detection ───────────────────────────────────────

_INSTITUTION_KEYWORDS = re.compile(
    r"""
    university|college|institute|department|dept\.|school|faculty|
    laboratory|lab\b|research|centre|center|corporation|corp\.|
    inc\.|ltd\.|google|meta|openai|microsoft|amazon|apple|deepmind|
    salesforce|\bai\b|@|\d{4,}|[{}[\]]|
    \bUSA\b|\bU\.S\.A\b|\bU\.K\b|\bChina\b|\bFrance\b|\bGermany\b|
    \bIndia\b|\bCanada\b|\bAustralia\b|\bJapan\b|\bKorea\b|\bItaly\b|
    \bSpain\b|\bNetherlands\b|\bSweden\b|\bSwitzerland\b|\bSingapore\b|
    \bBrazil\b|\bIsrael\b|\bPoland\b|\bDenmark\b|\bFinland\b|
    \bCUHK\b|\bHITSZ\b|\bKAIST\b|\bPOSTECH\b|\bINRIA\b|\bCNRS\b|
    \bETHz\b|\bEPFL\b|\bRIKEN\b|
    Huawei|Tencent|Alibaba|Baidu|ByteDance|Samsung|Sony|Toyota|
    \bIBM\b|\bNTT\b|\bNEC\b|\bFujitsu\b|
    Hong\s+Kong|Shenzhen|Beijing|Shanghai|Wuhan|Nanjing|Chengdu
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Detect lines that are purely institution abbreviations (e.g. "CUHK-Shenzhen, HITSZ, BIT")
_ABBREV_RE = re.compile(r"^[A-Z][A-Za-z0-9\-]{1,15}$")


def _is_institution_abbrev_line(text: str) -> bool:
    """Return True if every comma-token looks like an institution abbreviation."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) < 2:
        return False
    abbrev_count = sum(1 for p in parts if _ABBREV_RE.match(p) and p[0].isupper() and not p.islower())
    return abbrev_count >= len(parts) - 1  # allow 1 non-abbrev

# Lines that are clearly not author names
_SKIP_LINE_RE = re.compile(
    r"^\s*$|"                            # empty
    r"arxiv:|preprint|under review|"     # boilerplate
    r"^[*†‡§¶∗♢♦♠♥♣✉✝]+\s*$|"          # lone symbol lines
    r"https?://|doi:|code\s+http",       # URLs
    re.IGNORECASE,
)


def _is_author_line(line: str) -> bool:
    """Return True if the line plausibly contains author names."""
    if _SKIP_LINE_RE.search(line):
        return False
    if _INSTITUTION_KEYWORDS.search(line):
        return False
    if _is_institution_abbrev_line(line):
        return False
    # Must have at least one capital letter (names)
    if not re.search(r"[A-Z]", line):
        return False
    return True


def _clean_author_name(name: str) -> str:
    """Strip affiliation superscripts and trailing symbols."""
    # Remove superscript numbers and symbols: ¹²³⁴ 1,2 * † ‡ § ♢ ∗ ♦ ♠ ♥ ♣
    name = re.sub(r"[¹²³⁴⁵⁶⁷⁸⁹⁰*†‡§¶∗♢♦♠♥♣✉✝]+", "", name)
    name = re.sub(r"\d+(,\d+)*", "", name)  # remove numeric superscripts like 1,2
    return name.strip(" ,;")


# ── Year extraction ───────────────────────────────────────────────────────────

def _extract_year(markdown: str) -> Optional[int]:
    """Extract year from arXiv date stamp on first line."""
    first_line = markdown.split("\n")[0]
    # arXiv:XXXX.XXXXvN [cs.AI] DD Mon YYYY
    m = re.search(r"\b(20\d{2})\b", first_line)
    if m:
        return int(m.group(1))
    # Fallback: scan first 200 chars
    m = re.search(r"\b(20\d{2})\b", markdown[:200])
    return int(m.group(1)) if m else None


# ── Title extraction ──────────────────────────────────────────────────────────

def _extract_title(markdown: str) -> str:
    """First # header is the paper title."""
    for line in markdown.split("\n"):
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return "Unknown Title"


# ── Author extraction ─────────────────────────────────────────────────────────

def _extract_authors(markdown: str) -> list[str]:
    """
    Extract author names from the block between title and abstract.
    Handles both space/tab-separated (one line) and one-per-line formats.
    """
    lines = markdown.split("\n")
    in_author_block = False
    author_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Start collecting after the title header
        if stripped.startswith("#") and not in_author_block:
            title_text = stripped.lstrip("#").strip().lower()
            # Title found — start looking for authors on subsequent lines
            if title_text not in ("abstract", "keywords", "references"):
                in_author_block = True
            continue

        if not in_author_block:
            continue

        # A # header after the title: check if it's an author list (LlamaParse
        # sometimes promotes author lines to headers). If not, stop.
        if stripped.startswith("#"):
            header_text = stripped.lstrip("#").strip()
            header_lower = header_text.lower()
            if header_lower in ("abstract", "keywords", "references",
                                "introduction", "acm reference format:"):
                break
            # Author-like header: has commas and capital letters but no section words
            if "," in header_text and _is_author_line(header_text):
                author_lines.append(header_text)
            else:
                break
            continue

        if _is_author_line(stripped) and stripped:
            author_lines.append(stripped)
        elif author_lines:
            # First non-author line after we've found some — stop
            # (but only if we've passed at least one blank line or institution line)
            if not stripped:
                continue  # blank lines between author blocks are ok
            if _INSTITUTION_KEYWORDS.search(stripped):
                # Could be affiliations — keep going in case more authors follow
                continue
            break

    if not author_lines:
        return ["Unknown"]

    authors: list[str] = []
    for line in author_lines:
        # Try comma-separated first (handles "Author1, Author2, Author3")
        if "," in line:
            parts = [_clean_author_name(p) for p in line.split(",")]
        else:
            # Multiple authors space/tab separated on same line
            # Split on 2+ spaces or tabs
            parts = [_clean_author_name(p) for p in re.split(r"\s{2,}|\t", line)]

        authors.extend(p for p in parts if p and len(p) > 2)

    return authors if authors else ["Unknown"]


# ── Bibliography extraction ───────────────────────────────────────────────────

# Common reference entry patterns
_REF_START_RE = re.compile(
    r"^\s*\[(\d+)\]|"      # [1] style
    r"^\s*(\d+)\.\s+[A-Z]",  # 1. Author style
    re.MULTILINE,
)


def _extract_bibliography(markdown: str) -> list[str]:
    """
    Extract raw reference strings from the References section.
    Returns a list of strings, one per reference.
    """
    # Find the References section
    ref_match = re.search(
        r"^#+\s*references\b",
        markdown,
        re.IGNORECASE | re.MULTILINE,
    )
    if not ref_match:
        return []

    ref_text = markdown[ref_match.end():]

    # Stop at next major section if any
    next_section = re.search(r"^#+\s+\w", ref_text, re.MULTILINE)
    if next_section:
        ref_text = ref_text[: next_section.start()]

    # Try numbered format first ([1] or 1. Author)
    refs: list[str] = []
    current: list[str] = []

    for line in ref_text.split("\n"):
        if _REF_START_RE.match(line):
            if current:
                refs.append(" ".join(current).strip())
            current = [line.strip()]
        elif current and line.strip():
            current.append(line.strip())

    if current:
        refs.append(" ".join(current).strip())

    # Fallback: paragraph-separated author-year style (no numeric markers)
    if not refs:
        paragraphs = re.split(r"\n\s*\n", ref_text.strip())
        refs = [p.replace("\n", " ").strip() for p in paragraphs]

    return [r for r in refs if len(r) > 20]  # drop noise


# ── Public API ────────────────────────────────────────────────────────────────

def extract_paper_metadata(paper: ParsedPaper) -> dict:
    """Return {title, authors, year} extracted from the paper markdown."""
    return {
        "title": _extract_title(paper.markdown),
        "authors": _extract_authors(paper.markdown),
        "year": _extract_year(paper.markdown),
    }


def extract_bibliography(paper: ParsedPaper) -> Bibliography:
    """Return Bibliography with raw reference strings."""
    return Bibliography(
        paper_id=paper.paper_id,
        references=_extract_bibliography(paper.markdown),
    )
