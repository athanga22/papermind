"""
Trimodal retriever — dense + BM25 + graph, fused via Reciprocal Rank Fusion.
Optional Cohere cross-encoder reranking after fusion.

Each retrieval mode returns (chunk_id, score) pairs.
RRF merges them into a single ranked list.
Cohere rerank re-scores the RRF candidates with a cross-encoder for higher precision.

No LLM involved. One embedding API call per query.
"""

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import cohere
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from neo4j import GraphDatabase

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from ingestion.bm25_index import BM25Index

# ── Constants ─────────────────────────────────────────────────────────────────

EMBEDDING_MODEL     = "text-embedding-3-small"
COLLECTION_NAME     = "papers"
RRF_K               = 60       # standard RRF constant
TOP_K_EACH          = 20       # candidates per retrieval mode before fusion
TOP_K_FINAL         = 10       # final results returned (no rerank)
RERANK_CANDIDATES   = 20       # RRF candidates sent to Cohere cross-encoder
COHERE_RERANK_MODEL = "rerank-english-v3.0"


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    chunk_id:    str
    paper_id:    str
    paper_title: str
    section:     str
    text:        str
    score:       float
    sources:     list[str] = field(default_factory=list)


# ── Retriever ─────────────────────────────────────────────────────────────────

class TrimodalRetriever:
    """
    Query-time retriever. Instantiate once, call retrieve() per query.

    Supports three retrieval modes (mix-and-match via flags):
      - dense:   Qdrant vector search (text-embedding-3-small)
      - bm25:    BM25S sparse index
      - graph:   Neo4j entity-aware traversal (opt-in, precision-first)

    All active modes are fused via custom RRF.
    Optional Cohere cross-encoder rerank applied after fusion.

    Stage latencies for the last retrieve() call are stored in self.last_latencies.
    """

    def __init__(self) -> None:
        self._openai  = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._qdrant  = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"),
                                     port=int(os.getenv("QDRANT_PORT", "6333")))
        self._neo4j   = GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
        )
        self._bm25    = BM25Index()
        self._bm25.load()
        # Cohere is optional (only needed when rerank is enabled). Lazily init to
        # avoid requiring COHERE_API_KEY for pure retrieval runs.
        self._cohere: cohere.ClientV2 | None = None

        # Cache: chunk_id → payload (fetched lazily from Qdrant)
        self._payload_cache: dict[str, dict] = {}

        # Stage latencies from the most recent retrieve() call (ms)
        self.last_latencies: dict[str, float] = {}

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        resp = self._openai.embeddings.create(
            input=[text], model=EMBEDDING_MODEL
        )
        return resp.data[0].embedding

    # ── Dense retrieval ───────────────────────────────────────────────────────

    def _dense(self, query_vec: list[float], k: int) -> list[tuple[str, float]]:
        hits = self._qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=k,
            with_payload=True,
        ).points
        for h in hits:
            self._payload_cache[h.payload["chunk_id"]] = h.payload
        return [(h.payload["chunk_id"], h.score) for h in hits]

    # ── BM25 retrieval ────────────────────────────────────────────────────────

    def _bm25_search(self, query: str, k: int) -> list[tuple[str, float]]:
        return self._bm25.query(query, k=k)

    # ── Graph retrieval ───────────────────────────────────────────────────────

    def _graph(self, query: str, k: int) -> list[tuple[str, float]]:
        """
        Entity-aware graph retrieval with precision-first design.

        Strategy:
        1. Extract multi-word phrases AND individual significant terms from query
        2. Match entities by EXACT full name first (high precision), then prefix/suffix
        3. Prefer longer entity matches (more specific = more signal)
        4. Return top-k chunk_ids scored by weighted entity match count

        Design rationale: the old substring-contains match produced too many
        false positives (e.g. "rag" matching "storage", "graph" matching "paragraph").
        The new approach requires the entity name to be a meaningful subphrase of
        the query, reducing noise significantly.
        """
        stopwords = {
            "what", "which", "where", "when", "does", "that", "this", "with",
            "from", "have", "their", "used", "paper", "papers", "approach",
            "system", "model", "method", "using", "based", "architecture",
            "framework", "technique", "performance", "results", "data",
            "these", "those", "they", "them", "than", "then", "also",
            "such", "both", "each", "most", "more", "only", "into",
            "across", "between", "against", "within",
        }

        query_lower = query.lower()
        words = re.findall(r"\b\w+\b", query_lower)
        significant = [w for w in words if w not in stopwords and len(w) >= 4]
        if not significant:
            return []

        with self._neo4j.session() as session:
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE $q CONTAINS e.name
                  AND size(e.name) >= 4
                WITH e, size(e.name) AS name_len
                MATCH (p:Paper)-[r:MENTIONS]->(e)
                WHERE r.chunk_id IS NOT NULL
                RETURN r.chunk_id AS chunk_id,
                       sum(name_len) AS score
                ORDER BY score DESC
                LIMIT $k
                """,
                q=query_lower, k=k,
            )
            return [(row["chunk_id"], float(row["score"])) for row in result]

    # ── RRF fusion ────────────────────────────────────────────────────────────

    @staticmethod
    def _rrf(
        ranked_lists: list[list[tuple[str, float]]],
        labels: list[str],
        k: int = RRF_K,
    ) -> list[tuple[str, float, list[str]]]:
        """
        Reciprocal Rank Fusion — custom implementation, no library.

        Formula: score(d) = Σ  1 / (k + rank_i(d))
        over all lists i that contain document d.

        Returns (chunk_id, rrf_score, [source_labels]) sorted descending.
        """
        scores: dict[str, float]      = {}
        sources: dict[str, list[str]] = {}

        for label, ranked in zip(labels, ranked_lists):
            for rank, (chunk_id, _) in enumerate(ranked, start=1):
                scores[chunk_id]  = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
                sources.setdefault(chunk_id, []).append(label)

        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(cid, sc, sources[cid]) for cid, sc in fused]

    # ── Cohere cross-encoder rerank ───────────────────────────────────────────

    def _rerank(
        self,
        query: str,
        candidates: list["RetrievedChunk"],
        top_k: int,
    ) -> list["RetrievedChunk"]:
        """
        Cross-encoder reranking via Cohere rerank-english-v3.0.

        Sends up to RERANK_CANDIDATES chunks to Cohere, which scores each
        (query, chunk) pair with a cross-encoder — much more accurate than
        the bi-encoder cosine similarity used for initial retrieval.

        The Cohere relevance_score replaces the RRF score in the returned chunks.
        """
        if not candidates:
            return candidates

        if self._cohere is None:
            api_key = os.getenv("COHERE_API_KEY")
            if not api_key:
                # Rerank requested but Cohere not configured; degrade gracefully.
                return candidates[:top_k]
            self._cohere = cohere.ClientV2(api_key=api_key)

        docs = [c.text for c in candidates]

        # Retry with exponential backoff — trial keys are limited to 10 req/min.
        max_retries = 5
        wait = 7.0   # start at 7 s (safely under the 6 s/call trial limit)
        for attempt in range(max_retries):
            try:
                resp = self._cohere.rerank(
                    model=COHERE_RERANK_MODEL,
                    query=query,
                    documents=docs,
                    top_n=min(top_k, len(docs)),
                )
                break   # success
            except Exception as exc:
                if "429" in str(exc) or "TooManyRequests" in type(exc).__name__:
                    if attempt < max_retries - 1:
                        time.sleep(wait)
                        wait *= 2   # 7 → 14 → 28 → 56 s
                    else:
                        raise
                else:
                    raise

        reranked = []
        for r in resp.results:
            original = candidates[r.index]
            reranked.append(RetrievedChunk(
                chunk_id    = original.chunk_id,
                paper_id    = original.paper_id,
                paper_title = original.paper_title,
                section     = original.section,
                text        = original.text,
                score       = r.relevance_score,   # cross-encoder score, not RRF
                sources     = original.sources + ["cohere"],
            ))
        return reranked

    # ── Payload fetch ─────────────────────────────────────────────────────────

    def _fetch_payloads(self, chunk_ids: list[str]) -> None:
        missing = [cid for cid in chunk_ids if cid not in self._payload_cache]
        if not missing:
            return
        results, _ = self._qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter={"must": [{"key": "chunk_id",
                                     "match": {"any": missing}}]},
            limit=len(missing),
            with_payload=True,
            with_vectors=False,
        )
        for pt in results:
            self._payload_cache[pt.payload["chunk_id"]] = pt.payload

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_FINAL,
        use_dense:         bool = True,
        use_bm25:          bool = True,
        use_graph:         bool = False,
        use_rerank:        bool = False,
        rerank_candidates: int  = RERANK_CANDIDATES,
    ) -> list[RetrievedChunk]:
        """
        Run hybrid retrieval and return top_k fused results.

        Stage timings (ms) are stored in self.last_latencies after each call:
          embed_ms, dense_ms, bm25_ms, graph_ms (if enabled),
          rerank_ms (if enabled), fetch_ms

        Default: dense + BM25 (97% Hit@5, MRR=0.737).
        Graph is opt-in — historically dilutes RRF when entity coverage is sparse.
        Rerank is opt-in — adds ~300-600ms but improves precision.
        """
        self.last_latencies = {}

        # ── Embed query ───────────────────────────────────────────────────────
        t = time.perf_counter()
        query_vec = self._embed(query)
        self.last_latencies["embed_ms"] = (time.perf_counter() - t) * 1000

        ranked_lists: list[list[tuple[str, float]]] = []
        labels: list[str] = []

        # ── Dense retrieval ───────────────────────────────────────────────────
        if use_dense:
            t = time.perf_counter()
            dense_hits = self._dense(query_vec, TOP_K_EACH)
            self.last_latencies["dense_ms"] = (time.perf_counter() - t) * 1000
            ranked_lists.append(dense_hits)
            labels.append("dense")

        # ── BM25 retrieval ────────────────────────────────────────────────────
        if use_bm25:
            t = time.perf_counter()
            bm25_hits = self._bm25_search(query, TOP_K_EACH)
            self.last_latencies["bm25_ms"] = (time.perf_counter() - t) * 1000
            ranked_lists.append(bm25_hits)
            labels.append("bm25")

        # ── Graph retrieval ───────────────────────────────────────────────────
        if use_graph:
            t = time.perf_counter()
            graph_hits = self._graph(query, TOP_K_EACH)
            self.last_latencies["graph_ms"] = (time.perf_counter() - t) * 1000
            ranked_lists.append(graph_hits)
            labels.append("graph")

        # ── RRF fusion ────────────────────────────────────────────────────────
        # If reranking, pull a wider candidate pool so Cohere has more to work with.
        rrf_limit = rerank_candidates if use_rerank else top_k
        fused = self._rrf(ranked_lists, labels)[:rrf_limit]

        # ── Payload fetch ─────────────────────────────────────────────────────
        t = time.perf_counter()
        self._fetch_payloads([cid for cid, _, _ in fused])
        self.last_latencies["fetch_ms"] = (time.perf_counter() - t) * 1000

        candidates: list[RetrievedChunk] = []
        for chunk_id, score, srcs in fused:
            p = self._payload_cache.get(chunk_id)
            if not p:
                continue
            candidates.append(RetrievedChunk(
                chunk_id    = chunk_id,
                paper_id    = p["paper_id"],
                paper_title = p["paper_title"],
                section     = p["section"],
                text        = p["text"],
                score       = score,
                sources     = srcs,
            ))

        # ── Cohere rerank (optional) ──────────────────────────────────────────
        if use_rerank and candidates:
            t = time.perf_counter()
            results = self._rerank(query, candidates, top_k)
            self.last_latencies["rerank_ms"] = (time.perf_counter() - t) * 1000
        else:
            results = candidates[:top_k]

        return results

    def close(self) -> None:
        self._neo4j.close()

    def __enter__(self):  return self
    def __exit__(self, *_): self.close()
