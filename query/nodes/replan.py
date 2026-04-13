"""
Replan node for PaperMind (LangGraph).

This is intentionally minimal for now: it increments `replan_count` and
generates a fallback plan that retries retrieval with a single broad sub-query.

You can later replace this with a more sophisticated replanner (e.g., prompt an
LLM with failure signals, missing entities, and prior sub-queries).
"""

from __future__ import annotations

from typing import Any

from query.state import PaperMindState


def replan_node(state: PaperMindState) -> dict[str, Any]:
    replan_count = int(state.get("replan_count") or 0) + 1
    query = state.get("query") or ""

    # Fallback behavior: retry with the original query as a single sub-query.
    # This guarantees forward progress without requiring an LLM replanner yet.
    return {
        "replan_count": replan_count,
        "sub_queries": [query] if query else [],
        "failed_sub_queries": [],
    }

