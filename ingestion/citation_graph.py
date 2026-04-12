"""
Step 7 — Bibliography → citation graph (Neo4j).

Strategy:
  - Parse raw reference strings extracted in Step 3
  - Extract author surname + title fragment + year as a weak identifier
  - Write (:Paper)-[:CITES]->(:Reference {ref_text, year, authors_fragment}) nodes
  - If a Reference matches a Paper node already in the graph (by title similarity),
    also write a direct (:Paper)-[:CITES]->(:Paper) edge
  - Heuristic title matching: normalise whitespace/case, check substring overlap

This is intentionally lightweight — no DOI resolution, no external API calls.
The graph is useful for "papers that cite X" queries even with imperfect matching.

Graph additions:
  Nodes:
    (:Reference {ref_id, ref_text, year, authors_fragment, title_fragment})
  Edges:
    (:Paper)-[:CITES {ref_id}]->(:Reference)
    (:Paper)-[:CITES_PAPER]->(:Paper)   ← only when a cross-ref match is found
"""

import hashlib
import re
from typing import Optional

from neo4j import GraphDatabase
import os

from ingestion.models import Bibliography

# ── Reference parsing helpers ─────────────────────────────────────────────────

_YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")
_LEADING_MARKER_RE = re.compile(r"^\s*\[\d+\]\s*|^\s*\d+\.\s+")
_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "and", "or", "to",
    "with", "via", "from", "is", "are", "by", "at", "as", "its",
    "using", "based", "large", "language", "model", "models",
}


def _ref_id(paper_id: str, ref_text: str) -> str:
    """Stable short ID for a reference entry."""
    h = hashlib.md5((paper_id + ref_text[:80]).encode()).hexdigest()[:10]
    return f"ref_{h}"


def _extract_year(ref_text: str) -> Optional[int]:
    m = _YEAR_RE.search(ref_text)
    return int(m.group(1)) if m else None


def _title_fragment(ref_text: str) -> str:
    """
    Best-effort title extraction: take the longest quoted/capitalised phrase.
    Fallback: first 60 chars after stripping author preamble.
    """
    # Try to find text after year — often the title follows "Author(s). Year. Title."
    clean = _LEADING_MARKER_RE.sub("", ref_text)
    # Remove quoted strings first
    quoted = re.findall(r'"([^"]{10,})"', clean)
    if quoted:
        return quoted[0][:100]
    # Heuristic: after the first period following a year, grab up to next period
    m = _YEAR_RE.search(clean)
    if m:
        after_year = clean[m.end():].lstrip(". ")
        title = after_year.split(".")[0].strip()
        if len(title) > 15:
            return title[:100]
    # Fallback
    return clean[:60]


def _authors_fragment(ref_text: str) -> str:
    """First 80 chars of the cleaned ref — typically contains author names."""
    clean = _LEADING_MARKER_RE.sub("", ref_text)
    return clean[:80]


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


# ── Citation graph writer ─────────────────────────────────────────────────────

class CitationGraphWriter:
    """
    Writes citation edges from a Bibliography into Neo4j.
    """

    def __init__(self) -> None:
        self._driver = GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
        )

    def _fetch_paper_titles(self, session) -> dict[str, str]:
        """Return {paper_id: normalised_title} for all Paper nodes in graph."""
        result = session.run("MATCH (p:Paper) RETURN p.paper_id AS pid, p.title AS title")
        return {r["pid"]: _normalise(r["title"] or "") for r in result}

    def _find_matching_paper(
        self,
        title_frag: str,
        paper_titles: dict[str, str],
    ) -> Optional[str]:
        """
        Return paper_id with highest token overlap to title_frag (≥ 0.45 threshold).
        Uses Jaccard-style overlap on content words (stopwords removed).
        """
        norm_frag = _normalise(title_frag)
        if len(norm_frag) < 10:
            return None

        frag_words = set(norm_frag.split()) - _STOPWORDS
        if len(frag_words) < 3:
            return None

        best_pid, best_score = None, 0.0
        for pid, norm_title in paper_titles.items():
            title_words = set(norm_title.split()) - _STOPWORDS
            if not title_words:
                continue
            overlap = len(frag_words & title_words) / min(len(frag_words), len(title_words))
            if overlap > best_score:
                best_score = overlap
                best_pid = pid

        return best_pid if best_score >= 0.45 else None

    def write_bibliography(self, citing_paper_id: str, bib: Bibliography) -> int:
        """
        Write all references from bib as :Reference nodes + :CITES edges.
        Returns number of reference nodes written.
        """
        if not bib.references:
            return 0

        written = 0
        with self._driver.session() as session:
            paper_titles = self._fetch_paper_titles(session)

            for ref_text in bib.references:
                if len(ref_text.strip()) < 20:
                    continue

                rid = _ref_id(citing_paper_id, ref_text)
                year = _extract_year(ref_text)
                title_frag = _title_fragment(ref_text)
                authors_frag = _authors_fragment(ref_text)

                # Write Reference node + CITES edge
                session.run(
                    """
                    MERGE (r:Reference {ref_id: $ref_id})
                    SET r.ref_text = $ref_text,
                        r.year = $year,
                        r.title_fragment = $title_frag,
                        r.authors_fragment = $authors_frag
                    WITH r
                    MATCH (p:Paper {paper_id: $paper_id})
                    MERGE (p)-[:CITES {ref_id: $ref_id}]->(r)
                    """,
                    ref_id=rid,
                    ref_text=ref_text[:500],
                    year=year,
                    title_frag=title_frag,
                    authors_frag=authors_frag,
                    paper_id=citing_paper_id,
                )
                written += 1

                # If this ref matches one of our indexed papers → direct Paper-Paper edge
                matched_pid = self._find_matching_paper(title_frag, paper_titles)
                if matched_pid and matched_pid != citing_paper_id:
                    session.run(
                        """
                        MATCH (src:Paper {paper_id: $src_id})
                        MATCH (dst:Paper {paper_id: $dst_id})
                        MERGE (src)-[:CITES_PAPER]->(dst)
                        """,
                        src_id=citing_paper_id,
                        dst_id=matched_pid,
                    )

        return written

    def close(self) -> None:
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
