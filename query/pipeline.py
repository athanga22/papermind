"""
Phase 2 — Naive RAG query pipeline.

Simple linear pipeline: embed query → retrieve → synthesize.
No LangGraph, no re-planning, no memory — this is the baseline
everything else is measured against.

Phase 4 will replace this with a LangGraph agentic loop.

Usage:
    from query.pipeline import RAGPipeline
    pipeline = RAGPipeline()
    result = pipeline.run("What does SRAG append to chunks to improve retrieval?")
    print(result.answer)
    print(result.confidence)
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from query.retriever import RetrievedChunk, TrimodalRetriever
from query.synthesizer import Synthesizer, SynthesisResult

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_TOP_K   = 5    # chunks passed to synthesis (keep focused, not noisy)
RETRIEVAL_TOP_K = 10   # candidates from retriever before capping


# ── Pipeline output ───────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    query:       str
    answer:      str
    confidence:  str                              # high / medium / low
    chunks:      list[RetrievedChunk] = field(default_factory=list)
    citations:   list[str]            = field(default_factory=list)  # cited chunk_ids
    latency_ms:  float = 0.0


# ── Pipeline ──────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Phase 2 naive RAG pipeline: retrieve then synthesize.

    Instantiate once per session, call run() per query.
    Uses context manager for clean resource teardown.
    """

    def __init__(
        self,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self._retriever  = TrimodalRetriever()
        self._synthesizer = Synthesizer()
        self._top_k      = top_k

    def run(self, query: str) -> PipelineResult:
        """
        Run the full naive RAG pipeline for a single query.
        Returns a PipelineResult with answer, confidence, and source chunks.
        """
        t0 = time.perf_counter()

        # ── Step 1: Retrieve ──────────────────────────────────────────────────
        chunks = self._retriever.retrieve(
            query,
            top_k=RETRIEVAL_TOP_K,
            use_dense=True,
            use_bm25=True,
            use_graph=False,
        )
        # Cap to top-k for synthesis (more chunks = more noise + cost)
        synthesis_chunks = chunks[:self._top_k]

        # ── Step 2: Synthesize ────────────────────────────────────────────────
        result: SynthesisResult = self._synthesizer.synthesize(query, synthesis_chunks)

        latency_ms = (time.perf_counter() - t0) * 1000

        return PipelineResult(
            query       = query,
            answer      = result.answer,
            confidence  = result.confidence,
            chunks      = synthesis_chunks,
            citations   = result.citations,
            latency_ms  = latency_ms,
        )

    def close(self) -> None:
        self._retriever.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
