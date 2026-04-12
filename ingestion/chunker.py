"""
Step 2 — Section-aware chunking.

Strategy:
  1. Filter artifact headers (figure labels, diagram annotations, page numbers)
  2. Split markdown into sections on real # headers
  3. Within each section, extract tables as intact single chunks
  4. Split remaining text with SentenceSplitter (256 tokens, 20 overlap)
  5. Drop nodes shorter than 50 chars — empty/noise

No LLM calls during chunking. Table integrity preserved without MarkdownElementNodeParser
(which requires an LLM for table summarisation even when llm=None).
"""

import re
from typing import Optional

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from ingestion.models import Chunk, ParsedPaper

# ── Artifact header detection ─────────────────────────────────────────────────

# Headers matching these patterns are figure/table captions, diagram labels,
# or other non-section content that LlamaParse promoted to # level.
_ARTIFACT_PATTERNS = [
    r"^(figure|fig\.?)\s*\d+",
    r"^(table|tab\.?)\s*\d+",
    r"^(algorithm|alg\.?)\s*\d+",
    r"^(listing|lst\.?)\s*\d+",
    r"^equation\s*\d*$",
    r"^\(\d+\)$",           # equation numbers like (1), (2)
    r"^\d+\.?\s*$",         # lone page numbers: "1", "2."
    r"^[ivxlcdm]+\.?\s*$",  # roman numerals
]
_ARTIFACT_RE = re.compile(
    "|".join(_ARTIFACT_PATTERNS), re.IGNORECASE
)

# Single-word headers that are genuine academic sections — exempt from length filter
_KNOWN_SECTIONS = {
    "abstract", "introduction", "conclusion", "conclusions",
    "references", "bibliography", "acknowledgements", "acknowledgments",
    "keywords", "appendix", "appendices", "discussion", "methodology",
    "methods", "results", "evaluation", "experiments", "background",
    "motivation", "overview", "summary", "limitations", "contributions",
}

# Headers shorter than this are almost always labels, not real sections
_MIN_HEADER_WORDS = 2


def _is_artifact_header(header_text: str) -> bool:
    text = header_text.strip()
    # Whitelist: known academic single-word sections are never artifacts
    if text.lower() in _KNOWN_SECTIONS:
        return False
    if len(text.split()) < _MIN_HEADER_WORDS:
        return True
    return bool(_ARTIFACT_RE.match(text))


# ── Section splitting ─────────────────────────────────────────────────────────

def _split_into_sections(markdown: str) -> list[tuple[str, str]]:
    """
    Split markdown into (section_name, section_text) pairs.
    Artifact headers are demoted to plain text within the current section.
    """
    sections: list[tuple[str, str]] = []
    current_section = "preamble"
    current_lines: list[str] = []

    for line in markdown.split("\n"):
        if line.startswith("#"):
            header_text = line.lstrip("#").strip()
            if _is_artifact_header(header_text):
                # Demote to plain text — don't start a new section
                current_lines.append(header_text)
            else:
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append((current_section, body))
                current_section = header_text
                current_lines = []
        else:
            current_lines.append(line)

    # Flush last section
    body = "\n".join(current_lines).strip()
    if body:
        sections.append((current_section, body))

    return sections


# ── Table extraction ──────────────────────────────────────────────────────────

def _extract_tables(text: str) -> tuple[str, list[str]]:
    """
    Pull out contiguous table blocks (lines starting with |).
    Returns (text_without_tables, [table_strings]).
    Tables are kept intact — never split at row boundaries.
    """
    lines = text.split("\n")
    tables: list[str] = []
    text_lines: list[str] = []
    table_buf: list[str] = []

    for line in lines:
        if line.strip().startswith("|"):
            table_buf.append(line)
        else:
            if table_buf:
                tables.append("\n".join(table_buf))
                table_buf = []
            text_lines.append(line)

    if table_buf:
        tables.append("\n".join(table_buf))

    return "\n".join(text_lines), tables


# ── Chunker ───────────────────────────────────────────────────────────────────

class PaperChunker:
    """
    Converts a ParsedPaper (markdown) into a list of Chunks.
    Requires paper-level metadata (title, authors, year) from the caller —
    these are injected in Step 3.
    """

    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 20) -> None:
        self._splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @staticmethod
    def _chunk_id(paper_id: str, index: int) -> str:
        return f"{paper_id}_{index:04d}"

    @staticmethod
    def _has_math(text: str) -> bool:
        return bool(re.search(r"\$[^$\n]+\$|\$\$[\s\S]+?\$\$", text))

    def chunk(
        self,
        paper: ParsedPaper,
        title: str,
        authors: list[str],
        year: Optional[int],
    ) -> list[Chunk]:
        """
        Chunk a single paper into indexable units.
        Returns chunks ordered by position in the paper.
        """
        sections = _split_into_sections(paper.markdown)
        chunks: list[Chunk] = []
        idx = 0

        for section_name, section_text in sections:
            text_body, tables = _extract_tables(section_text)

            # ── Table chunks (one per table, kept intact) ──────────────────
            for table_md in tables:
                if len(table_md.strip()) < 50:
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=self._chunk_id(paper.paper_id, idx),
                        paper_id=paper.paper_id,
                        paper_title=title,
                        authors=authors,
                        year=year,
                        section=section_name,
                        chunk_index=idx,
                        text=table_md.strip(),
                        is_table=True,
                        contains_math=self._has_math(table_md),
                    )
                )
                idx += 1

            # ── Text chunks (split by SentenceSplitter) ────────────────────
            text_body = text_body.strip()
            if not text_body:
                continue

            doc = Document(text=text_body)
            nodes = self._splitter.get_nodes_from_documents([doc])

            for node in nodes:
                text = node.text.strip()
                if len(text) < 50:
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=self._chunk_id(paper.paper_id, idx),
                        paper_id=paper.paper_id,
                        paper_title=title,
                        authors=authors,
                        year=year,
                        section=section_name,
                        chunk_index=idx,
                        text=text,
                        is_table=False,
                        contains_math=self._has_math(text),
                    )
                )
                idx += 1

        return chunks
