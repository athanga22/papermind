"""
LangGraph state for PaperMind.

`PaperMindState` is the central state object passed between LangGraph nodes.

Reducer notes:
  - `retrieved_chunks`: uses a custom reducer that handles two cases:
      (a) parallel fan-out branches → appends (operator.add behaviour)
      (b) rerank_node writes back a replacement list → replaces entirely
      The rerank node signals replacement by wrapping its list in a sentinel dict:
      {"__replace__": True, "chunks": [...]}
  - `failed_sub_queries`: operator.add (parallel branches only, never replaced)
"""

from typing import TypedDict, List, Annotated, Any
import operator


def _chunks_reducer(existing: list, update: Any) -> list:
    """
    Custom reducer for retrieved_chunks.

    - If `update` is a list → append (parallel branch fan-in, operator.add semantics)
    - If `update` is {"__replace__": True, "chunks": [...]} → replace entirely
      (used by rerank_node to write back the reranked + capped list)
    """
    if isinstance(update, dict) and update.get("__replace__"):
        return list(update.get("chunks", []))
    if isinstance(update, list):
        return list(existing or []) + list(update)
    return list(existing or [])


def _latencies_reducer(existing: dict, update: dict) -> dict:
    """Merge stage_latencies dicts. Later writes overwrite earlier ones (last-write wins)."""
    return {**(existing or {}), **(update or {})}


class PaperMindState(TypedDict):
    query: str
    # Set by classifier_node (Adaptive-RAG pattern). Planner reads this as a
    # hard cap on sub-query count. Values: 2 (simple), 4 (comparison),
    # 4 (moderate), 5 (synthesis).
    max_sub_queries: int
    # Set by classifier_node for comparison queries: the paper titles (as they
    # appear in the corpus) that the question explicitly names. Planner uses
    # these as lexical anchors to force per-paper sub-queries so retrieval
    # covers both papers instead of whichever has denser keyword overlap.
    target_papers: List[str]
    sub_queries: List[str]
    retrieved_chunks: Annotated[List[dict], _chunks_reducer]
    # failed_sub_queries is also populated from parallel branches.
    failed_sub_queries: Annotated[List[str], operator.add]
    replan_count: int
    synthesis: str
    confidence_score: float
    # Per-node wall-clock timings in ms. Each node merges its own entry.
    # retrieve_ms = slowest parallel branch (≈ fan-out wall time).
    stage_latencies: Annotated[dict, _latencies_reducer]
