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

from query.state import PaperMindState


# Agent synthesis uses Sonnet — cross-paper synthesis is reasoning-heavy.
# Override with PAPERMIND_SYNTHESIS_MODEL env var if needed.
SYNTHESIS_MODEL = os.getenv("PAPERMIND_SYNTHESIS_MODEL", "claude-haiku-4-5-20251001")
MAX_TOKENS = int(os.getenv("PAPERMIND_SYNTHESIS_MAX_TOKENS", "900"))
# Hard cap on chunks sent to synthesis. With 5 chunks per sub-query and up to 6
# sub-queries, deduped chunks can still reach 20-30. Beyond ~20, synthesis quality
# degrades — too much noise, answer relevancy drops.
MAX_SYNTHESIS_CHUNKS = int(os.getenv("PAPERMIND_SYNTHESIS_MAX_CHUNKS", "20"))


_SYSTEM = """\
You are PaperMind, a precise research synthesis assistant.

You MUST answer using ONLY the provided <doc> sources.

Citations:
- Every factual claim must be supported by at least one citation.
- Cite using [doc N] where N is the id attribute of the <doc>.
- If sources do not support the answer, say:
  "I don't have enough information in the provided sources to answer this question reliably."

At the end, output a JSON object on its own line:
{"confidence": "high"|"medium"|"low"}
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


def _strip_confidence_json(raw: str) -> str:
    return re.sub(r'\s*\{"confidence":\s*"(high|medium|low)"\}\s*$', "", raw).strip()


def _parse_confidence(raw: str) -> str:
    m = re.search(r'\{"confidence":\s*"(high|medium|low)"\}', raw)
    return m.group(1) if m else "medium"


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


def synthesis_node(state: PaperMindState) -> dict[str, Any]:
    query = state.get("query") or ""
    raw_chunks = list(state.get("retrieved_chunks") or [])
    # rerank_node already deduped, reranked, and capped at RERANK_TOP_N.
    # _dedup_chunks here is a safety net in case rerank_node was bypassed.
    chunks = _dedup_chunks(raw_chunks)[:MAX_SYNTHESIS_CHUNKS]
    if not chunks:
        return {
            "synthesis": "I don't have enough information in the provided sources to answer this question reliably."
        }

    docs = _build_docs(chunks)
    user = (
        f"<question>{query}</question>\n\n"
        f"<sources>\n{docs}\n</sources>\n\n"
        "Write a concise, well-structured answer. "
        "Cite each claim inline using [doc N]."
    )

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=SYNTHESIS_MODEL,
        max_tokens=MAX_TOKENS,
        system=_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )

    raw = resp.content[0].text
    answer = _strip_confidence_json(raw)
    # Keep confidence JSON requirement enforced, but only store synthesis in state for now.
    _ = _parse_confidence(raw)

    return {"synthesis": answer}

