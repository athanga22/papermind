"""
Confidence gate for PaperMind (LangGraph).

Purpose
- Compute a lightweight confidence score from retrieval results.
- Route: if confidence < threshold -> replan, else -> synthesis.

This intentionally uses a deterministic heuristic (no LLM) so it's fast, cheap,
and easy to calibrate.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any, Iterable

from query.state import PaperMindState


DEFAULT_THRESHOLD = float(os.getenv("PAPERMIND_GATE_THRESHOLD", "0.6"))
MIN_TOTAL_CHUNKS = int(os.getenv("PAPERMIND_GATE_MIN_CHUNKS", "3"))


_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than",
    "to", "of", "in", "on", "for", "with", "from", "by", "as", "at", "into",
    "is", "are", "was", "were", "be", "being", "been", "do", "does", "did",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "compare", "comparison", "versus", "vs",
    "explain", "describe", "summarize", "overview",
    "paper", "papers", "method", "model", "approach", "system",
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-]{1,}", text)


def extract_core_entities(query: str) -> list[str]:
    """
    Heuristic "core entities" extractor.

    We bias toward terms likely to be stable anchors across retrieval:
    - ALLCAPS tokens (RAG, SRAG, MCTS)
    - CamelCase / TitleCase-ish tokens (UniRoute, xRouter)
    - tokens with digits (GPT-4o, LlamaParse)
    - longer content words (len>=4) excluding stopwords
    """
    raw = _tokens(query)
    keep: list[str] = []
    for t in raw:
        low = t.lower()
        if low in _STOPWORDS:
            continue
        if len(t) >= 4:
            keep.append(t)
            continue
        # short tokens: only keep if they look like acronyms/identifiers
        if t.isupper() and len(t) >= 2:
            keep.append(t)
        elif re.search(r"\d", t):
            keep.append(t)
        elif re.search(r"[A-Z].*[A-Z]", t):  # e.g. "xRouter" or "UniRoute"
            keep.append(t)

    # de-dup while preserving order
    seen = set()
    out: list[str] = []
    for t in keep:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def _mention_coverage(entities: Iterable[str], haystack: str) -> float:
    ents = list(entities)
    if not ents:
        return 0.0
    h = haystack.lower()
    hit = 0
    for e in ents:
        if e.lower() in h:
            hit += 1
    return hit / len(ents)


def compute_confidence(state: PaperMindState) -> float:
    """
    Compute confidence in [0,1] from the retrieved chunks.

    Intuition:
    - If core query entities are repeatedly mentioned in retrieved contexts,
      retrieval is likely on-target -> higher confidence.
    - Penalize low evidence (few total chunks).
    - Penalize sub-queries that retrieve context that doesn't mention the core entities.
    """
    query = state.get("query") or ""
    entities = extract_core_entities(query)
    chunks = list(state.get("retrieved_chunks") or [])
    if not chunks:
        return 0.0

    # Group chunks by sub_query if present (planner/map stage adds this).
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in chunks:
        grouped[str(c.get("sub_query", ""))].append(c)

    # Per-subquery coverage: how many entities appear in the retrieved text set.
    per_sq_scores: list[float] = []
    for sq, items in grouped.items():
        text_blob = "\n".join(
            str(it.get("paper_title", "")) + "\n" + str(it.get("section", "")) + "\n" + str(it.get("text", ""))
            for it in items
        )
        per_sq_scores.append(_mention_coverage(entities, text_blob))

    # Aggregate: average coverage across sub-queries (or chunks if sub_query absent).
    coverage = sum(per_sq_scores) / len(per_sq_scores) if per_sq_scores else 0.0

    # Evidence factor: encourage having enough total chunks
    evidence = min(1.0, len(chunks) / max(1, MIN_TOTAL_CHUNKS))

    # Final blended score (simple + stable)
    score = 0.8 * coverage + 0.2 * evidence
    return max(0.0, min(1.0, score))


def gate_node(state: PaperMindState) -> dict[str, Any]:
    """LangGraph node: compute and store `confidence_score`."""
    return {"confidence_score": compute_confidence(state)}


def gate_route(state: PaperMindState, threshold: float = DEFAULT_THRESHOLD) -> str:
    """
    LangGraph conditional router.

    Returns:
      - "replan" if confidence is below threshold
      - "synthesize" otherwise
    """
    score = float(state.get("confidence_score") or 0.0)
    return "replan" if score < threshold else "synthesize"

