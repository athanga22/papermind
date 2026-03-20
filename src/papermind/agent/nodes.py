"""
LangGraph node functions.

Each function takes AgentState and returns a partial state dict.
"""

from __future__ import annotations

import json
import logging

import anthropic

from ..config import settings
from ..ingestion.models import RetrievedChunk
from ..retrieval.embedder import get_embedder
from ..retrieval.fusion import reciprocal_rank_fusion
from ..retrieval.reranker import get_reranker
from ..retrieval.sparse import BM25Index
from ..retrieval.store import get_store
from .prompts import (
    GENERATOR_HUMAN,
    GENERATOR_SYSTEM,
    HALLUCINATION_HUMAN,
    HALLUCINATION_SYSTEM,
    REWRITER_HUMAN,
    REWRITER_SYSTEM,
    ROUTER_HUMAN,
    ROUTER_SYSTEM,
)
from .state import AgentState

logger = logging.getLogger(__name__)

# ── Module-level singletons ───────────────────────────────────────────────────
_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
_bm25: BM25Index | None = None  # Lazily initialised


def _get_bm25() -> BM25Index:
    global _bm25
    if _bm25 is None or not _bm25.chunks:
        _bm25 = BM25Index()
        all_chunks = get_store().get_all_chunks()
        _bm25.build(all_chunks)
    return _bm25


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a numbered context string for the LLM."""
    parts: list[str] = []
    for i, rc in enumerate(chunks, start=1):
        c = rc.chunk
        header = f"[{i}] {c.paper_title} | {c.section.title()} | p.{c.page_number}"
        parts.append(f"{header}\n{c.text}")
    return "\n\n---\n\n".join(parts)


def _claude(system: str, human: str) -> str:
    """Single Claude API call — returns text content."""
    msg = _client.messages.create(
        model=settings.claude_model,
        max_tokens=settings.claude_max_tokens,
        temperature=settings.claude_temperature,
        system=system,
        messages=[{"role": "user", "content": human}],
    )
    return msg.content[0].text.strip()  # type: ignore[union-attr]


# ── Node: route_query ─────────────────────────────────────────────────────────

def route_query(state: AgentState) -> dict:  # type: ignore[type-arg]
    """Classify the question as 'retrieval' or 'conversational'."""
    question = state["question"]
    try:
        result = _claude(
            system=ROUTER_SYSTEM,
            human=ROUTER_HUMAN.format(question=question),
        ).lower()
        query_type = "retrieval" if "retrieval" in result else "conversational"
    except Exception as e:
        logger.warning("Router failed (%s), defaulting to retrieval", e)
        query_type = "retrieval"

    logger.info("Query type: %s", query_type)
    return {
        "query_type": query_type,
        "active_query": question,
        "retry_count": 0,
    }


# ── Node: retrieve ────────────────────────────────────────────────────────────

def retrieve(state: AgentState) -> dict:  # type: ignore[type-arg]
    """Hybrid search (dense + BM25) → RRF fusion → cross-encoder reranking."""
    query = state["active_query"]
    store = get_store()
    bm25 = _get_bm25()
    reranker = get_reranker()

    # Dense search
    dense_results = store.search(query, top_k=settings.dense_top_k)

    # Sparse (BM25) search
    sparse_results = bm25.search(query, top_k=settings.sparse_top_k)

    # RRF fusion
    fused = reciprocal_rank_fusion(
        [dense_results, sparse_results],
        top_k=settings.dense_top_k,  # keep more for reranker input
    )

    # Cross-encoder reranking → final top-K
    reranked = reranker.rerank(query, fused, top_k=settings.rerank_top_k)
    top_score = reranker.top_score(reranked)

    logger.info(
        "Retrieval: %d dense + %d sparse → %d fused → %d reranked (top score: %.3f)",
        len(dense_results),
        len(sparse_results),
        len(fused),
        len(reranked),
        top_score,
    )

    return {
        "retrieved_chunks": reranked,
        "top_rerank_score": top_score,
    }


# ── Node: check_confidence ────────────────────────────────────────────────────

def check_confidence(state: AgentState) -> dict:  # type: ignore[type-arg]
    """
    Decide whether to proceed to generation or rewrite the query.

    Triggers rewrite if:
      - top rerank score < threshold, AND
      - haven't already retried (retry_count < 1)
    """
    score = state["top_rerank_score"]
    retry_count = state["retry_count"]
    should_rewrite = score < settings.reranker_threshold and retry_count < 1

    logger.info(
        "Confidence check: score=%.3f threshold=%.3f retry=%d → rewrite=%s",
        score,
        settings.reranker_threshold,
        retry_count,
        should_rewrite,
    )
    return {"should_rewrite": should_rewrite}


# ── Node: rewrite_query ───────────────────────────────────────────────────────

def rewrite_query(state: AgentState) -> dict:  # type: ignore[type-arg]
    """Rewrite the query to improve retrieval recall."""
    original = state["question"]
    try:
        rewritten = _claude(
            system=REWRITER_SYSTEM,
            human=REWRITER_HUMAN.format(question=original),
        )
    except Exception as e:
        logger.warning("Query rewrite failed (%s), using original", e)
        rewritten = original

    logger.info("Rewritten query: %r → %r", original, rewritten)
    return {
        "active_query": rewritten,
        "retry_count": state["retry_count"] + 1,
    }


# ── Node: generate ────────────────────────────────────────────────────────────

def generate(state: AgentState) -> dict:  # type: ignore[type-arg]
    """Generate a citation-grounded answer using Claude."""
    question = state["question"]
    chunks = state.get("retrieved_chunks", [])
    query_type = state.get("query_type", "retrieval")

    if query_type == "conversational":
        # Lightweight conversational response — no retrieval context
        answer = _claude(
            system="You are a helpful assistant for a research paper Q&A system.",
            human=question,
        )
        return {"answer": answer, "citations": [], "is_grounded": True, "grounding_issues": []}

    if not chunks:
        return {
            "answer": "I couldn't find relevant information in the indexed papers to answer this question.",
            "citations": [],
            "is_grounded": True,
            "grounding_issues": [],
        }

    context = _format_context(chunks)
    answer = _claude(
        system=GENERATOR_SYSTEM,
        human=GENERATOR_HUMAN.format(question=question, context=context),
    )

    citations = [
        {
            "paper_title": rc.chunk.paper_title,
            "section": rc.chunk.section,
            "page_number": rc.chunk.page_number,
            "score": rc.score,
            "rank": rc.rank,
            "text_preview": rc.chunk.text[:200],
        }
        for rc in chunks
    ]

    return {"answer": answer, "citations": citations}


# ── Node: check_hallucination ─────────────────────────────────────────────────

def check_hallucination(state: AgentState) -> dict:  # type: ignore[type-arg]
    """Verify the generated answer is grounded in the retrieved context."""
    answer = state.get("answer", "")
    chunks = state.get("retrieved_chunks", [])

    if not chunks or not answer:
        return {"is_grounded": True, "grounding_issues": []}

    context = _format_context(chunks)
    try:
        raw = _claude(
            system=HALLUCINATION_SYSTEM,
            human=HALLUCINATION_HUMAN.format(answer=answer, context=context),
        )
        # Strip potential markdown code fences
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        result = json.loads(raw)
        is_grounded: bool = result.get("is_grounded", True)
        issues: list[str] = result.get("issues", [])
    except Exception as e:
        logger.warning("Hallucination check failed (%s), assuming grounded", e)
        is_grounded, issues = True, []

    if not is_grounded:
        logger.warning("Grounding issues detected: %s", issues)

    return {"is_grounded": is_grounded, "grounding_issues": issues}
