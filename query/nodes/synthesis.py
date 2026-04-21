"""
Synthesis node for PaperMind (LangGraph).

Takes the combined retrieved chunks from prior nodes and produces a grounded,
cited answer. Uses Anthropic Claude Haiku by default.

Input (from state):
  - query: str
  - retrieved_chunks: list[dict] with keys like paper_title, section, text

Output (state update):
  - synthesis: str
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import anthropic
from langgraph.config import get_stream_writer

from query.state import PaperMindState


# Sonnet for cross-paper reasoning (comparison/synthesis/contradiction).
# Haiku for simple/moderate — factoids and methodology questions don't need Sonnet.
# Faithfulness on cross-paper comparisons dropped to 0.776 with Haiku alone.
SYNTHESIS_MODEL = os.getenv("PAPERMIND_SYNTHESIS_MODEL", "claude-sonnet-4-5-20250929")
SYNTHESIS_MODEL_FAST = os.getenv("PAPERMIND_SYNTHESIS_MODEL_FAST", "claude-haiku-4-5-20251001")
# 450 tokens ≈ 300-350 words — enough for a well-cited comparative answer.
# Raising this to 900 inflated Sonnet synthesis latency to 14-15s; 450 targets ~7s.
MAX_TOKENS = int(os.getenv("PAPERMIND_SYNTHESIS_MAX_TOKENS", "450"))
# Hard cap on chunks sent to synthesis. With 5 chunks per sub-query and up to 6
# sub-queries, deduped chunks can still reach 20-30. Beyond ~20, synthesis quality
# degrades — too much noise, answer relevancy drops.
MAX_SYNTHESIS_CHUNKS = int(os.getenv("PAPERMIND_SYNTHESIS_MAX_CHUNKS", "20"))


_SYSTEM = """\
You are PaperMind, a precise research synthesis assistant.

Your PRIMARY obligation is to directly answer the specific question asked. \
Do not summarize the retrieved sources — use them as evidence to answer the question. \
Structure your response around what the question is asking, not around what the documents say.

You MUST answer using ONLY the provided <doc> sources.

Citations:
- Every factual claim must be supported by at least one citation.
- Cite using [doc N] where N is the id attribute of the <doc>.
- If sources do not support the answer, say:
  "I don't have enough information in the provided sources to answer this question reliably."
"""


def _build_docs(chunks: list[dict[str, Any]], limit: int | None = None) -> str:
    items = chunks[:limit] if limit else chunks
    parts: list[str] = []
    for i, c in enumerate(items, 1):
        title = str(c.get("paper_title", "")).strip() or "Unknown"
        section = str(c.get("section", "")).strip() or "Unknown"
        text = str(c.get("text", "")).strip()
        parts.append(
            f'<doc id="{i}" title="{title}" section="{section}">\n'
            f"{text}\n"
            f"</doc>"
        )
    return "\n\n".join(parts)


# Phrases the LLM uses when it cannot ground an answer in the retrieved sources.
_ABSTENTION_PHRASES = (
    "don't have enough information",
    "do not have enough information",
    "cannot determine",
    "not mentioned in the provided",
    "no information in the provided",
    "unable to find in",
    "insufficient information",
)


def _compute_confidence(chunks: list[dict[str, Any]], answer: str) -> float:
    """
    Derive confidence from two real signals — no LLM self-report needed.

    Signal 1 — Abstention language (overrides everything):
      The LLM is instructed to say "I don't have enough information..." when
      sources don't support the answer. Detecting this phrase is more reliable
      than asking the LLM to self-rate.

    Signal 2 — Cohere rerank score (primary retrieval signal):
      Cohere scores (0–1) measure how relevant the top-ranked chunk is to the
      query. Linear mapping: 0.0→0.30, 0.5→0.625, 1.0→0.95. This gives
      high (≥0.8) only when the retriever found genuinely relevant content.

    Fallback — RRF-only (no Cohere):
      RRF scores are not calibrated for absolute thresholding. Default to a
      neutral 0.65 (medium) when Cohere wasn't used.
    """
    if any(p in answer.lower() for p in _ABSTENTION_PHRASES):
        return 0.2

    if not chunks:
        return 0.2

    has_cohere = any("cohere" in c.get("sources", []) for c in chunks)
    if has_cohere:
        max_score = max(c.get("score", 0.0) for c in chunks)
        return round(min(0.3 + max_score * 0.65, 0.95), 4)

    return 0.65 if len(chunks) >= 5 else 0.5


def _dedup_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Deduplicate chunks by chunk_id.

    Multiple sub-queries may retrieve the same chunk. Keep the first occurrence
    (highest-score, since retrieval nodes return descending-score results).
    """
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for c in chunks:
        cid = str(c.get("chunk_id", id(c)))
        if cid not in seen:
            seen.add(cid)
            out.append(c)
    return out


def _pick_model(state: PaperMindState) -> str:
    """
    Route to Haiku for simple/moderate queries, Sonnet for cross-paper work.

    Inference logic (no extra state field needed):
      max_sub_queries==2               → simple     → Haiku
      max_sub_queries==4, no targets   → moderate   → Haiku
      max_sub_queries==4, has targets  → comparison → Sonnet
      max_sub_queries==5               → synthesis  → Sonnet
    """
    max_sq = int(state.get("max_sub_queries") or 4)
    has_targets = bool(state.get("target_papers"))
    if max_sq <= 2 or (max_sq == 4 and not has_targets):
        return SYNTHESIS_MODEL_FAST
    return SYNTHESIS_MODEL


def synthesis_node(state: PaperMindState) -> dict[str, Any]:
    query = state.get("query") or ""
    raw_chunks = list(state.get("retrieved_chunks") or [])
    # rerank_node already deduped, reranked, and capped at RERANK_TOP_N.
    # _dedup_chunks here is a safety net in case rerank_node was bypassed.
    chunks = _dedup_chunks(raw_chunks)[:MAX_SYNTHESIS_CHUNKS]
    if not chunks:
        return {
            "synthesis": "I don't have enough information in the provided sources to answer this question reliably.",
            "confidence_score": 0.3,
        }

    model = _pick_model(state)
    docs = _build_docs(chunks)
    user = (
        f"<question>{query}</question>\n\n"
        f"<sources>\n{docs}\n</sources>\n\n"
        "Write a direct, well-structured answer in 3 paragraphs or fewer (under 250 words). "
        "Cite each claim inline using [doc N]."
    )

    write = get_stream_writer()
    write({"type": "progress", "node": "synthesize", "message": "Synthesizing answer..."})

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    parts: list[str] = []

    with client.messages.stream(
        model=model,
        max_tokens=MAX_TOKENS,
        system=_SYSTEM,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        for text in stream.text_stream:
            write({"type": "token", "content": text})
            parts.append(text)

    answer = "".join(parts).strip()
    confidence_score = _compute_confidence(chunks, answer)

    return {"synthesis": answer, "confidence_score": confidence_score}

