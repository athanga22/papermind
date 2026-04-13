"""
Trimodal retriever — dense + BM25 + graph, fused via Reciprocal Rank Fusion.

Each retrieval mode returns (chunk_id, score) pairs.
RRF merges them into a single ranked list.

No LLM involved. One embedding API call per query.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from neo4j import GraphDatabase

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from ingestion.bm25_index import BM25Index

# ── Constants ─────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME  = "papers"
RRF_K            = 60       # standard RRF constant
TOP_K_EACH       = 20       # candidates per retrieval mode before fusion
TOP_K_FINAL      = 10       # final results returned


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    chunk_id:    str
    paper_id:    str
    paper_title: str
    section:     str
    text:        str
    score:       float
    sources:     list[str] = field(default_factory=list)  # which modes found it


# ── Retriever ─────────────────────────────────────────────────────────────────

class TrimodalRetriever:
    """
    Query-time retriever. Instantiate once, call retrieve() per query.
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

        # Cache: chunk_id → payload (fetched lazily)
        self._payload_cache: dict[str, dict] = {}

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
        # Extended stopwords — terms that appear in entity names but carry no query signal
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

        # Quick bail-out: if no significant words, graph can't help
        words = re.findall(r"\b\w+\b", query_lower)
        significant = [w for w in words if w not in stopwords and len(w) >= 4]
        if not significant:
            return []

        with self._neo4j.session() as session:
            # Match entities whose FULL name appears verbatim in the query.
            # query CONTAINS e.name  (reversed from old logic — entity IS a substring of query)
            # This means: only return chunks about entities the user explicitly mentioned.
            # Name length scoring favours specific entities (e.g. "bert" > "be").
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
        Reciprocal Rank Fusion.
        Returns (chunk_id, rrf_score, [source_labels]) sorted desc.
        """
        scores: dict[str, float]      = {}
        sources: dict[str, list[str]] = {}

        for label, ranked in zip(labels, ranked_lists):
            for rank, (chunk_id, _) in enumerate(ranked, start=1):
                scores[chunk_id]  = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
                sources.setdefault(chunk_id, []).append(label)

        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(cid, sc, sources[cid]) for cid, sc in fused]

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
        use_dense: bool = True,
        use_bm25:  bool = True,
        use_graph: bool = False,
    ) -> list[RetrievedChunk]:
        """
        Run hybrid retrieval and return top_k fused results.

        Default: dense + BM25 (97% Hit@5, MRR=0.737).
        Graph is opt-in — it returns disjoint results that dilute RRF fusion,
        dropping Hit@5 from 97% → 90% when enabled.
        """
        query_vec = self._embed(query)

        ranked_lists, labels = [], []

        if use_dense:
            dense_hits = self._dense(query_vec, TOP_K_EACH)
            ranked_lists.append(dense_hits)
            labels.append("dense")

        if use_bm25:
            bm25_hits = self._bm25_search(query, TOP_K_EACH)
            ranked_lists.append(bm25_hits)
            labels.append("bm25")

        if use_graph:
            graph_hits = self._graph(query, TOP_K_EACH)
            ranked_lists.append(graph_hits)
            labels.append("graph")

        fused = self._rrf(ranked_lists, labels)[:top_k]

        # Fetch payloads for any chunk not already cached
        self._fetch_payloads([cid for cid, _, _ in fused])

        results = []
        for chunk_id, score, srcs in fused:
            p = self._payload_cache.get(chunk_id)
            if not p:
                continue
            results.append(RetrievedChunk(
                chunk_id    = chunk_id,
                paper_id    = p["paper_id"],
                paper_title = p["paper_title"],
                section     = p["section"],
                text        = p["text"],
                score       = score,
                sources     = srcs,
            ))
        return results

    def close(self) -> None:
        self._neo4j.close()

    def __enter__(self):  return self
    def __exit__(self, *_): self.close()
