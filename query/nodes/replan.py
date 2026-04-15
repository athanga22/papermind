"""
Replan node for PaperMind (LangGraph).

When the confidence gate determines retrieval quality is insufficient, this node
calls Claude Sonnet to generate alternative sub-queries based on:
  - The original user query
  - The prior sub-queries that failed
  - The confidence score that triggered replan

The LLM replanner is significantly better than the old fallback (which just
re-ran the original query verbatim). It reframes the question with different
vocabulary, more specific scopes, and alternative angles.

Max replan budget is enforced in gate.py — this node is only called when
budget remains.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import anthropic
from langgraph.config import get_stream_writer

from query.state import PaperMindState


REPLAN_MODEL = os.getenv("PAPERMIND_REPLAN_MODEL", "claude-haiku-4-5-20251001")


_SYSTEM = """You are the PaperMind Retrieval Replanner. A previous retrieval attempt failed to find sufficient evidence.

Your task: generate NEW, DIFFERENT sub-queries that approach the same information need from a different angle.

**Rules:**
- Output ONLY a valid JSON list of strings. No preamble, no markdown, no explanations.
- Format: ["query 1", "query 2", "query 3"]
- Do NOT repeat the exact sub-queries that already failed.
- Use different vocabulary, synonyms, and angles.
- Try more specific technical terms OR broader section anchors (e.g. "abstract", "conclusion", "results table").
- Each sub-query should be 5-15 words.
- Generate 2-4 sub-queries (fewer is fine if 2 strong angles exist).
"""


def _parse_json_list(text: str) -> list[str]:
    try:
        data = json.loads(text)
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return [s.strip() for s in data if s.strip()]
    except Exception:
        pass
    m = re.search(r"\[[\s\S]*\]", text)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        pass
    return []


def replan_node(state: PaperMindState) -> dict[str, Any]:
    get_stream_writer()({"type": "progress", "node": "replan", "message": "Retrieval insufficient — replanning..."})
    replan_count = int(state.get("replan_count") or 0) + 1
    query = state.get("query") or ""
    prior_sub_queries = list(state.get("sub_queries") or [])
    failed_sub_queries = list(state.get("failed_sub_queries") or [])
    confidence = float(state.get("confidence_score") or 0.0)

    prior_str = "\n".join(f"- {sq}" for sq in prior_sub_queries) if prior_sub_queries else "(none)"
    failed_str = "\n".join(f"- {sq}" for sq in failed_sub_queries) if failed_sub_queries else "(none)"

    user = (
        f"Original user question: {query}\n\n"
        f"Prior sub-queries that were tried (do NOT repeat these):\n{prior_str}\n\n"
        f"Sub-queries that hard-failed (retrieval error):\n{failed_str}\n\n"
        f"Confidence score achieved: {confidence:.2f} (threshold 0.6 — too low)\n\n"
        "Generate alternative sub-queries that approach the same information need differently."
    )

    try:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        resp = client.messages.create(
            model=REPLAN_MODEL,
            max_tokens=300,
            system=_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        new_sub_queries = _parse_json_list(resp.content[0].text)
    except Exception:
        # Graceful fallback: use the original query as a single broad search
        new_sub_queries = [query] if query else []

    # If LLM returned nothing useful, fall back to original query
    if not new_sub_queries:
        new_sub_queries = [query] if query else []

    return {
        "replan_count": replan_count,
        "sub_queries": new_sub_queries,
        # Reset retrieved_chunks so the new retrieval pass starts fresh.
        # LangGraph operator.add will append, so we cannot zero it here —
        # the new chunks will be appended on top of existing ones and synthesis
        # will see the full union (which is desirable: more context, not less).
        "failed_sub_queries": [],
    }
