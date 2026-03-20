"""LangGraph state definition for the PaperMind agent."""

from __future__ import annotations

from typing import Any, TypedDict

from ..ingestion.models import RetrievedChunk


class AgentState(TypedDict):
    # Input
    question: str

    # Routing
    query_type: str  # "retrieval" | "conversational"

    # Retrieval
    active_query: str              # Current query (may be rewritten)
    retrieved_chunks: list[RetrievedChunk]
    top_rerank_score: float

    # Control flow
    retry_count: int               # Max 1 rewrite retry
    should_rewrite: bool

    # Output
    answer: str
    citations: list[dict[str, Any]]  # Serialisable citation objects
    is_grounded: bool
    grounding_issues: list[str]

    # Error handling
    error: str | None
