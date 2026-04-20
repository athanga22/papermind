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


# Synthesis uses claude-sonnet-4-5 — faithfulness on cross-paper comparisons
# dropped to 0.776 with Haiku (over-inference on partial context). Sonnet is
# more conservative about citing claims it can't ground. claude-sonnet-4-5-20250929
# is the cheaper sonnet tier; override with PAPERMIND_SYNTHESIS_MODEL if needed.
SYNTHESIS_MODEL = os.getenv("PAPERMIND_SYNTHESIS_MODEL", "claude-sonnet-4-5-20250929")
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
        "Write a direct, well-structured answer in 3 paragraphs or fewer (under 250 words). "
        "Cite each claim inline using [doc N]."
    )

    write = get_stream_writer()
    write({"type": "progress", "node": "synthesize", "message": "Synthesizing answer..."})

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    parts: list[str] = []

    # Use Anthropic's streaming API so tokens are forwarded to the SSE stream
    # in real time. Falls back gracefully to a no-op writer when called from
    # run_agent() (eval / non-streaming path) — get_stream_writer() returns a
    # no-op in that context.
    with client.messages.stream(
        model=SYNTHESIS_MODEL,
        max_tokens=MAX_TOKENS,
        system=_SYSTEM,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        for text in stream.text_stream:
            write({"type": "token", "content": text})
            parts.append(text)

    raw    = "".join(parts)
    answer = _strip_confidence_json(raw)
    _      = _parse_confidence(raw)

    return {"synthesis": answer}

