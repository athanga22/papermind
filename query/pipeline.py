"""
Phase 2/3 — Naive RAG query pipeline with optional Cohere reranking.

Linear pipeline: embed query → hybrid retrieve → [optional rerank] → synthesize.
No LangGraph, no re-planning, no memory — this is the baseline
everything else is measured against.

Phase 4 will replace this with a LangGraph agentic loop.

Usage:
    # Baseline (dense + BM25 + RRF, no rerank)
    with RAGPipeline() as pipeline:
        result = pipeline.run("What does SRAG append to chunks to improve retrieval?")

    # Step 2: with Cohere reranking
    with RAGPipeline(use_rerank=True) as pipeline:
        result = pipeline.run("What does SRAG append to chunks to improve retrieval?")
        print(result.stage_latencies)  # {"embed_ms": X, "dense_ms": X, "rerank_ms": X, ...}

    # Step 3: with graph traversal
    with RAGPipeline(use_graph=True) as pipeline:
        result = pipeline.run("How does LONGMEM relate to MemoryBank?")
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from query.retriever import RetrievedChunk, TrimodalRetriever
from query.synthesizer import Synthesizer, SynthesisResult, SYNTHESIS_MODEL
from query.tracing import get_client as get_tracer

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_TOP_K     = 5    # chunks passed to synthesis (keep focused, not noisy)
RETRIEVAL_TOP_K   = 10   # candidates from retriever when not reranking
RERANK_CANDIDATES = 20   # wider pool sent to Cohere cross-encoder before top-k cut


# ── Pipeline output ───────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    query:           str
    answer:          str
    confidence:      str                              # high / medium / low
    chunks:          list[RetrievedChunk] = field(default_factory=list)
    citations:       list[str]            = field(default_factory=list)
    latency_ms:      float = 0.0
    stage_latencies: dict  = field(default_factory=dict)
    # stage_latencies keys (whichever are active):
    #   embed_ms, dense_ms, bm25_ms, graph_ms, fetch_ms, rerank_ms, synthesize_ms


# ── Pipeline ──────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Phase 2/3 RAG pipeline: retrieve → [rerank] → synthesize.

    Instantiate once per session, call run() per query.
    Uses context manager for clean resource teardown.

    Args:
        top_k:           Final chunks sent to synthesis (default 5).
        synthesis_model: Claude model for answer generation.
        use_rerank:      Enable Cohere cross-encoder reranking (Step 2).
        use_graph:       Enable Neo4j graph traversal as third retrieval path (Step 3).
    """

    def __init__(
        self,
        top_k: int = DEFAULT_TOP_K,
        synthesis_model: str | None = None,
        use_rerank: bool = False,
        use_graph:  bool = False,
    ) -> None:
        self._retriever   = TrimodalRetriever()
        self._synthesizer = Synthesizer(model=synthesis_model or SYNTHESIS_MODEL)
        self._top_k       = top_k
        self._use_rerank  = use_rerank
        self._use_graph   = use_graph

    def run(self, query: str, session_id: str | None = None) -> PipelineResult:
        """
        Run the full RAG pipeline for a single query.

        Retrieval candidate pool:
          - No rerank: top RETRIEVAL_TOP_K from RRF, capped to top_k for synthesis.
          - Rerank:    top RERANK_CANDIDATES from RRF → Cohere → top_k for synthesis.

        Returns PipelineResult with answer, confidence, source chunks, and
        per-stage latencies (ms).

        Langfuse trace structure (when tracing is configured):
          Agent  "rag-query"
            ├── Retriever "retrieval"   — chunk_ids, papers, stage latencies
            └── Generation "synthesis" — model, prompt, answer, token counts
        """
        t0 = time.perf_counter()
        lf = get_tracer()

        mode_tags = ["hybrid"]
        if self._use_rerank: mode_tags.append("rerank")
        if self._use_graph:  mode_tags.append("graph")

        # ── Retrieve ──────────────────────────────────────────────────────────
        if lf:
            with lf.start_as_current_observation(
                name="retrieval",
                as_type="retriever",
                input={"query":      query,
                       "use_rerank": self._use_rerank,
                       "use_graph":  self._use_graph},
            ) as ret_span:
                chunks = self._retriever.retrieve(
                    query,
                    top_k             = self._top_k,
                    use_dense         = True,
                    use_bm25          = True,
                    use_graph         = self._use_graph,
                    use_rerank        = self._use_rerank,
                    rerank_candidates = RERANK_CANDIDATES,
                )
                stage_latencies = dict(self._retriever.last_latencies)
                ret_span.update(
                    output={
                        "n_chunks":  len(chunks),
                        "chunk_ids": [c.chunk_id for c in chunks],
                        "papers":    list({c.paper_title for c in chunks}),
                        "sources":   [c.sources for c in chunks],
                    },
                    metadata=stage_latencies,
                )
        else:
            chunks = self._retriever.retrieve(
                query,
                top_k             = self._top_k,
                use_dense         = True,
                use_bm25          = True,
                use_graph         = self._use_graph,
                use_rerank        = self._use_rerank,
                rerank_candidates = RERANK_CANDIDATES,
            )
            stage_latencies = dict(self._retriever.last_latencies)

        # ── Synthesize ────────────────────────────────────────────────────────
        t_synth = time.perf_counter()
        if lf:
            with lf.start_as_current_observation(
                name="synthesis",
                as_type="generation",
                model=self._synthesizer._model,
                input={
                    "query":    query,
                    "contexts": [
                        {"chunk_id": c.chunk_id,
                         "paper":    c.paper_title,
                         "section":  c.section,
                         "text":     c.text[:400]}
                        for c in chunks
                    ],
                },
            ) as syn_span:
                result = self._synthesizer.synthesize(query, chunks)
                syn_span.update(
                    output=result.answer,
                    usage_details={
                        "input":  result.input_tokens,
                        "output": result.output_tokens,
                    },
                    metadata={"confidence": result.confidence},
                )
        else:
            result = self._synthesizer.synthesize(query, chunks)

        stage_latencies["synthesize_ms"] = (time.perf_counter() - t_synth) * 1000
        latency_ms = (time.perf_counter() - t0) * 1000

        return PipelineResult(
            query           = query,
            answer          = result.answer,
            confidence      = result.confidence,
            chunks          = chunks,
            citations       = result.citations,
            latency_ms      = latency_ms,
            stage_latencies = stage_latencies,
        )

    def close(self) -> None:
        self._retriever.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
