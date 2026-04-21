"""
Confidence gate for PaperMind (LangGraph).

Uses a rerank-score threshold instead of an LLM call.

Why score threshold instead of LLM:
  - The original Haiku LLM gate (~1.5s, ~$0.002/query) fired on only 4/116 eval
    queries. It cost 174s of aggregate latency to catch 4 clear misses.
  - Cohere's reranker already scores every chunk (0-1 relevance). A chunk with
    relevance > 0.1 means the retriever found something topically relevant.
    Threshold check: 0ms, 0 tokens.
  - When Cohere is not active (RRF scores in 0.01-0.09 range), we default to
    sufficient=True — RRF scores are not calibrated for absolute thresholding.
  - Hard replan cap (MAX_REPLAN_ATTEMPTS) still enforces forward progress.
"""

from __future__ import annotations

import os
from typing import Any

from langgraph.config import get_stream_writer

from query.state import PaperMindState

MAX_REPLAN_ATTEMPTS = int(os.getenv("PAPERMIND_MAX_REPLANS", "2"))

# Minimum Cohere relevance score to consider retrieval sufficient.
# Cohere scores range 0-1; below 0.1 means no chunk has meaningful relevance.
COHERE_SCORE_THRESHOLD = float(os.getenv("PAPERMIND_GATE_THRESHOLD", "0.1"))


def _score_sufficient(chunks: list[dict[str, Any]]) -> bool:
    if not chunks:
        return False
    has_cohere = any("cohere" in c.get("sources", []) for c in chunks)
    if has_cohere:
        max_score = max(c.get("score", 0.0) for c in chunks)
        return max_score >= COHERE_SCORE_THRESHOLD
    # RRF scores — not calibrated for absolute thresholding, default sufficient.
    return True


def gate_node(state: PaperMindState) -> dict[str, Any]:
    get_stream_writer()({"type": "progress", "node": "gate", "message": "Checking retrieval quality..."})
    chunks = list(state.get("retrieved_chunks") or [])
    sufficient = _score_sufficient(chunks)
    return {"confidence_score": 1.0 if sufficient else 0.0}


def gate_route(state: PaperMindState) -> str:
    replan_count = int(state.get("replan_count") or 0)
    if replan_count >= MAX_REPLAN_ATTEMPTS:
        return "synthesize"
    score = float(state.get("confidence_score") or 0.0)
    return "replan" if score < 0.5 else "synthesize"
