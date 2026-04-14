"""
Post-pool reranking node for PaperMind (LangGraph).

After all sub-queries retrieve chunks in parallel, the pool is deduplicated and
reranked against the ORIGINAL question (not individual sub-queries) using Cohere's
cross-encoder.

Why this matters (Anthropic Contextual Retrieval post, NVIDIA reranking blog):
  - RRF scores are sub-query-relative. A chunk ranked #1 for sub-query A and a
    chunk ranked #1 for sub-query B are not directly comparable — they were scored
    against different queries.
  - A cross-encoder scores all pooled chunks against a single query string (the
    original question) in one pass, producing a unified, comparable ranking.
  - Anthropic's experiments: contextual embeddings + BM25 + reranking reduces
    retrieval failures by 67% vs. contextual embeddings alone.
  - The literature consensus: retrieve 50-150 candidates broadly, rerank to top 20.

This node runs AFTER all retrieve_one branches complete and BEFORE the gate judge.
The gate judge sees a clean, reranked top-20 rather than a noisy 40-90 chunk pool.

If COHERE_API_KEY is not set, this node falls back to top-20 by RRF score (same
as before). The gate and synthesis still work correctly either way.
"""

from __future__ import annotations

import os
import time
from typing import Any

from query.state import PaperMindState

RERANK_TOP_N = int(os.getenv("PAPERMIND_RERANK_TOP_N", "20"))
COHERE_RERANK_MODEL = "rerank-english-v3.0"


def _dedup(chunks: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for c in chunks:
        cid = str(c.get("chunk_id", id(c)))
        if cid not in seen:
            seen.add(cid)
            out.append(c)
    return out


def _fallback_top_n(chunks: list[dict], n: int) -> list[dict]:
    """Fallback: sort deduped chunks by descending RRF score, take top n."""
    return sorted(chunks, key=lambda c: float(c.get("score", 0.0)), reverse=True)[:n]


def rerank_node(state: PaperMindState) -> dict[str, Any]:
    """
    LangGraph node: deduplicate and rerank the full retrieval pool against the
    original question, then cap at RERANK_TOP_N chunks.

    Writes back to `retrieved_chunks` — downstream nodes (gate, synthesis) see
    only the top-ranked chunks.
    """
    query = state.get("query") or ""
    raw_chunks = list(state.get("retrieved_chunks") or [])
    deduped = _dedup(raw_chunks)

    if not deduped:
        return {"retrieved_chunks": {"__replace__": True, "chunks": []}}

    api_key = os.getenv("COHERE_API_KEY", "")
    if not api_key:
        top_n = _fallback_top_n(deduped, RERANK_TOP_N)
        return {"retrieved_chunks": {"__replace__": True, "chunks": top_n}}

    try:
        import cohere
        co = cohere.ClientV2(api_key=api_key)
        docs = [str(c.get("text", "")) for c in deduped]

        max_retries = 3
        wait = 7.0
        for attempt in range(max_retries):
            try:
                resp = co.rerank(
                    model=COHERE_RERANK_MODEL,
                    query=query,
                    documents=docs,
                    top_n=min(RERANK_TOP_N, len(docs)),
                )
                break
            except Exception as exc:
                if "429" in str(exc) and attempt < max_retries - 1:
                    time.sleep(wait)
                    wait *= 2
                else:
                    raise

        reranked: list[dict] = []
        for r in resp.results:
            original = deduped[r.index]
            reranked.append({**original, "score": r.relevance_score, "sources": original.get("sources", []) + ["cohere"]})

        # Use sentinel to signal replacement (not append) to the custom reducer
        return {"retrieved_chunks": {"__replace__": True, "chunks": reranked}}

    except Exception:
        # Rerank failure is non-fatal — fall back to top-N by RRF score
        top_n = _fallback_top_n(deduped, RERANK_TOP_N)
        return {"retrieved_chunks": {"__replace__": True, "chunks": top_n}}
