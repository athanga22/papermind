"""
Qdrant vector store wrapper.

Handles collection management, upserting chunks, and dense similarity search.
BM25 lives in-memory (sparse.py); Qdrant owns the dense index and the
canonical chunk payload store.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ..config import settings
from ..ingestion.models import Chunk, RetrievedChunk
from .embedder import Embedder, get_embedder

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(
        self,
        url: str = settings.qdrant_url,
        collection: str = settings.qdrant_collection,
        embedder: Embedder | None = None,
    ) -> None:
        self._client = QdrantClient(url=url, timeout=30)
        self._collection = collection
        self._embedder = embedder or get_embedder()

    # ── Collection management ─────────────────────────────────────────────

    def ensure_collection(self, recreate: bool = False) -> None:
        """Create the collection if it doesn't exist (or recreate it)."""
        existing = {c.name for c in self._client.get_collections().collections}

        if recreate and self._collection in existing:
            logger.warning("Recreating collection '%s'", self._collection)
            self._client.delete_collection(self._collection)
            existing.discard(self._collection)

        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qmodels.VectorParams(
                    size=self._embedder.dim,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            # Payload indexes for metadata filtering
            for field_name, schema in (
                ("paper_id", qmodels.PayloadSchemaType.KEYWORD),
                ("section", qmodels.PayloadSchemaType.KEYWORD),
                ("year", qmodels.PayloadSchemaType.INTEGER),
                ("contains_math", qmodels.PayloadSchemaType.BOOL),
            ):
                self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field_name,
                    field_schema=schema,
                )
            logger.info("Created collection '%s'", self._collection)

    # ── Ingestion ─────────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        """Embed and upsert chunks in batches."""
        if not chunks:
            return

        texts = [c.text for c in chunks]
        vectors = self._embedder.embed_batch(texts, batch_size=batch_size)

        points = [
            qmodels.PointStruct(
                id=c.chunk_id,
                vector=vec,
                payload=c.to_payload(),
            )
            for c, vec in zip(chunks, vectors)
        ]

        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self._client.upsert(collection_name=self._collection, points=batch)

        logger.info("Upserted %d chunks into '%s'", len(chunks), self._collection)

    # ── Retrieval ─────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = settings.dense_top_k,
        filter_: qmodels.Filter | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Dense similarity search. Returns (chunk, score) pairs sorted desc."""
        query_vec = self._embedder.embed(query)
        response = self._client.query_points(
            collection_name=self._collection,
            query=query_vec,
            limit=top_k,
            query_filter=filter_,
            with_payload=True,
        )
        return [
            (Chunk.from_payload(str(r.id), r.payload or {}), r.score)
            for r in response.points
        ]

    def get_all_chunks(self) -> list[Chunk]:
        """
        Scroll through all chunks — used to rebuild the BM25 index on startup.
        For large collections this should be paginated; fine for portfolio scale.
        """
        chunks: list[Chunk] = []
        offset = None

        while True:
            records, offset = self._client.scroll(
                collection_name=self._collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for rec in records:
                if rec.payload:
                    chunks.append(Chunk.from_payload(str(rec.id), rec.payload))
            if offset is None:
                break

        logger.info("Loaded %d chunks from Qdrant for BM25 rebuild", len(chunks))
        return chunks

    def count(self) -> int:
        return self._client.count(collection_name=self._collection).count


@lru_cache(maxsize=1)
def get_store() -> VectorStore:
    """Module-level singleton."""
    return VectorStore()
