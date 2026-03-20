"""
LangGraph state machine.

Flow:
  START
    → route_query
    → [conversational] → generate → check_hallucination → END
    → [retrieval]      → retrieve → check_confidence
                                  → [confident]     → generate → check_hallucination → END
                                  → [needs rewrite] → rewrite_query → retrieve (once)
                                                    → generate → check_hallucination → END
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from .nodes import (
    check_confidence,
    check_hallucination,
    generate,
    retrieve,
    rewrite_query,
    route_query,
)
from .state import AgentState

logger = logging.getLogger(__name__)


def _route_after_routing(state: AgentState) -> str:
    return state["query_type"]


def _route_after_confidence(state: AgentState) -> str:
    return "rewrite" if state["should_rewrite"] else "generate"


def build_graph() -> StateGraph:  # type: ignore[type-arg]
    graph: StateGraph = StateGraph(AgentState)  # type: ignore[type-arg]

    graph.add_node("route_query", route_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("check_confidence", check_confidence)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate", generate)
    graph.add_node("check_hallucination", check_hallucination)

    graph.add_edge(START, "route_query")

    graph.add_conditional_edges(
        "route_query",
        _route_after_routing,
        {
            "retrieval": "retrieve",
            "conversational": "generate",
        },
    )

    graph.add_edge("retrieve", "check_confidence")

    graph.add_conditional_edges(
        "check_confidence",
        _route_after_confidence,
        {
            "rewrite": "rewrite_query",
            "generate": "generate",
        },
    )

    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", "check_hallucination")
    graph.add_edge("check_hallucination", END)

    return graph


# Compiled app — import this in UI and eval code
app = build_graph().compile()


def run(question: str) -> AgentState:
    """Convenience wrapper — run the agent and return the final state."""
    initial: AgentState = {
        "question": question,
        "query_type": "",
        "active_query": question,
        "retrieved_chunks": [],
        "top_rerank_score": 0.0,
        "retry_count": 0,
        "should_rewrite": False,
        "answer": "",
        "citations": [],
        "is_grounded": True,
        "grounding_issues": [],
        "error": None,
    }
    result: AgentState = app.invoke(initial)  # type: ignore[assignment]
    return result
