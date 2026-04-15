"""
PaperMind LangGraph agent — v2 wiring.

Graph (Adaptive-RAG pattern):
  classifier -> planner -> fan-out retrieve (parallel) -> rerank -> gate
  gate -> {replan | synthesize}
  replan -> planner  (re-plans with new sub-queries, not a full restart)
  synthesize -> END

Key changes from v1:
  - classifier_node (Haiku, ~0.2s) classifies query complexity before planning.
    Routes budget: simple=2, moderate=4, complex=6 sub-queries.
  - rerank_node sits between retrieval fan-in and the gate.
    Deduplicates the full pool and reranks against the ORIGINAL question
    using Cohere cross-encoder (or falls back to top-20 by RRF score).
  - gate_node now uses an LLM sufficiency judge (Haiku) instead of the
    entity-overlap heuristic that failed on cross-paper questions.
  - replan routes back to planner (not dispatch_retrieval), so the planner
    can generate fresh sub-queries with different angles.

Entry point:
  run_agent(query) -> dict with keys: synthesis, confidence_score, replan_count,
                                       sub_queries, retrieved_chunks, cache_hit
"""

from __future__ import annotations

import functools
import time
from pathlib import Path
from typing import Any, List

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from langgraph.types import Send

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from query.nodes.classifier import classifier_node
from query.nodes.gate import gate_node, gate_route
from query.nodes.planner import planner_node
from query.nodes.replan import replan_node
from query.nodes.rerank import rerank_node
from query.nodes.retrieval import retrieve_one_node
from query.nodes.synthesis import synthesis_node
from query.state import PaperMindState
from query.tracing import get_client as get_tracer


def _timed(name: str, fn):
    """
    Wrap a LangGraph node function with wall-clock timing.
    Records `{name}_ms` into `stage_latencies` in the returned state update.
    For parallel nodes (retrieve_one), all branches write `retrieve_ms` —
    last-write wins, which approximates the fan-out wall time.
    """
    @functools.wraps(fn)
    def wrapper(state):
        t0 = time.perf_counter()
        result = fn(state)
        ms = round((time.perf_counter() - t0) * 1000, 1)
        if isinstance(result, dict):
            existing = result.get("stage_latencies") or {}
            result["stage_latencies"] = {**existing, f"{name}_ms": ms}
        return result
    return wrapper


def dispatch_retrieval(state: PaperMindState) -> List[Send]:
    """
    Fan-out dispatcher: one parallel retrieve_one per sub-query.
    Used as a routing function in conditional_edges (not a node).

    LangGraph executes all Send() branches concurrently (thread-pool).
    For a 4-sub-query comparison (2 per paper), all 4 retrievals run in
    parallel — paper A and paper B are fetched simultaneously, not serially.
    Retrieved chunks are merged by the _chunks_reducer in state.py before
    rerank_node runs.
    """
    sub_queries = list(state.get("sub_queries") or [])
    return [Send("retrieve_one", {"sub_queries": [sq]}) for sq in sub_queries]


def build_app():
    workflow = StateGraph(PaperMindState)

    workflow.add_node("classifier", _timed("classifier", classifier_node))
    workflow.add_node("planner",    _timed("planner",    planner_node))
    workflow.add_node("retrieve_one", _timed("retrieve", retrieve_one_node))
    workflow.add_node("rerank",     _timed("rerank",     rerank_node))
    workflow.add_node("gate",       _timed("gate",       gate_node))
    workflow.add_node("replan",     _timed("replan",     replan_node))
    workflow.add_node("synthesize", _timed("synthesis",  synthesis_node))

    # classifier -> planner -> fan-out retrieve
    workflow.set_entry_point("classifier")
    workflow.add_edge("classifier", "planner")
    workflow.add_conditional_edges("planner", dispatch_retrieval, ["retrieve_one"])

    # All retrieve_one branches converge at rerank (via operator.add on retrieved_chunks)
    workflow.add_edge("retrieve_one", "rerank")

    # rerank -> gate -> route
    workflow.add_edge("rerank", "gate")
    workflow.add_conditional_edges(
        "gate",
        gate_route,
        {"replan": "replan", "synthesize": "synthesize"},
    )

    # replan -> planner (re-plan with new sub-queries, then retrieve again)
    workflow.add_edge("replan", "planner")
    workflow.add_edge("synthesize", END)

    return workflow.compile()


app = build_app()


def initial_state_for(query: str) -> PaperMindState:
    """Return a clean initial state dict for the given query."""
    return {
        "query":              query,
        "max_sub_queries":    4,
        "target_papers":      [],
        "sub_queries":        [],
        "retrieved_chunks":   [],
        "failed_sub_queries": [],
        "replan_count":       0,
        "synthesis":          "",
        "confidence_score":   0.0,
        "stage_latencies":    {},
    }


def run_agent(
    query: str,
    session_id: str | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """
    Run the PaperMind agentic pipeline for a single query.

    Checks semantic cache first (use_cache=True by default).
    On miss, runs the full LangGraph pipeline and stores the result.
    """
    # ── Semantic cache check ──────────────────────────────────────────────
    cache = None
    if use_cache:
        try:
            from query.cache import SemanticCache
            cache = SemanticCache()
            cached = cache.get(query)
            if cached is not None:
                cached["cache_hit"] = True
                return cached
        except Exception:
            cache = None

    # ── Full pipeline run ─────────────────────────────────────────────────
    initial_state: PaperMindState = initial_state_for(query)

    lf = get_tracer()
    t0 = time.perf_counter()

    if lf:
        with lf.start_as_current_observation(
            name="agent-query",
            as_type="agent",
            input={"query": query},
            metadata={"session_id": session_id},
        ) as span:
            final_state = app.invoke(initial_state)
            latency_ms = (time.perf_counter() - t0) * 1000
            span.update(
                output={
                    "synthesis": final_state.get("synthesis", ""),
                    "confidence_score": final_state.get("confidence_score", 0.0),
                    "replan_count": final_state.get("replan_count", 0),
                    "sub_queries": final_state.get("sub_queries", []),
                    "max_sub_queries": final_state.get("max_sub_queries", 4),
                    "n_chunks": len(final_state.get("retrieved_chunks", [])),
                },
                metadata={"latency_ms": latency_ms, "session_id": session_id},
            )
        lf.flush()
    else:
        final_state = app.invoke(initial_state)

    final_state["cache_hit"] = False

    if cache is not None:
        try:
            cache.put(query, final_state)
            cache.flush()
        except Exception:
            pass

    return final_state
