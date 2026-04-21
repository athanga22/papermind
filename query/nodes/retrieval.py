"""
Retrieval nodes for PaperMind (LangGraph).

LangGraph parallelism pattern (recommended):
- Map: run `retrieve_one_node` once per `sub_query` in parallel via LangGraph.
- Reduce: rely on state merge (`operator.add`) for `retrieved_chunks`.

Note: Cohere reranking is intentionally disabled here (agent-only delta).
"""

from __future__ import annotations

import os
import threading
from typing import Any

from langgraph.config import get_stream_writer

from query.retriever import TrimodalRetriever
from query.state import PaperMindState


RETRIEVAL_TOP_K = int(os.getenv("PAPERMIND_RETRIEVAL_TOP_K", "5"))

# Singleton retriever — initialized once, shared across all parallel retrieve_one
# branches. Avoids opening a new Qdrant connection + loading the BM25 index on
# every sub-query. close() only shuts down Neo4j (unused), so sharing is safe.
_RETRIEVER: TrimodalRetriever | None = None
_RETRIEVER_LOCK = threading.Lock()


def _get_retriever() -> TrimodalRetriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        with _RETRIEVER_LOCK:
            if _RETRIEVER is None:
                _RETRIEVER = TrimodalRetriever()
    return _RETRIEVER


def _chunk_to_dict(chunk, sub_query: str) -> dict[str, Any]:
    return {
        "sub_query": sub_query,
        "chunk_id": chunk.chunk_id,
        "paper_id": chunk.paper_id,
        "paper_title": chunk.paper_title,
        "section": chunk.section,
        "text": chunk.text,
        "score": chunk.score,
        "sources": list(chunk.sources),
    }


def retrieve_one_node(state: PaperMindState) -> dict[str, Any]:
    """
    LangGraph *map* node: retrieve chunks for a single `sub_query`.

    Expected input:
      - state["query"]: str (optional, unused here)
      - state["sub_queries"]: list[str] (we read the first entry)

    Returns a *partial* state update (do not return the full state):
      - {"retrieved_chunks": [ ... ]} on success
      - {"failed_sub_queries": [sub_query]} on failure

    Why this shape:
      - `retrieved_chunks` is annotated with `operator.add` in `PaperMindState`,
        so LangGraph can merge results from parallel branches by appending.
    """
    sub_queries = list(state.get("sub_queries") or [])
    if not sub_queries:
        return {"retrieved_chunks": [], "failed_sub_queries": []}

    sub_query = sub_queries[0]
    get_stream_writer()({"type": "progress", "node": "retrieve", "message": f"Retrieving: {sub_query[:60]}"})

    try:
        retriever = _get_retriever()
        chunks = retriever.retrieve(
            sub_query,
            top_k=RETRIEVAL_TOP_K,
            use_dense=True,
            use_bm25=True,
            use_graph=False,
            use_rerank=False,
        )
        return {
            "retrieved_chunks": [_chunk_to_dict(c, sub_query) for c in chunks],
            "failed_sub_queries": [],
        }
    except Exception:
        return {
            "retrieved_chunks": [],
            "failed_sub_queries": [sub_query],
        }


def retrieval_reduce_node(state: PaperMindState) -> dict[str, Any]:
    """
    Optional reducer node.

    In most LangGraph setups you can skip an explicit reducer because
    `retrieved_chunks` merges via `operator.add`. This node exists as a hook for
    later deduping / trimming policies.
    """
    return {}

