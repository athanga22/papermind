"""
Qdrant vector store — Step 1: dense search only.

Handles collection setup, upserting chunks, and cosine similarity search.
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

    def ensure_collection(self, recreate: bool = False) -> None:
        """Create the collection if it doesn't exist, or recreate it."""
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
            # Index fields we'll filter on later
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

    def upsert_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        """Embed and upsert chunks into Qdrant in batches."""
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
            self._client.upsert(
                collection_name=self._collection,
                points=points[i : i + batch_size],
            )

        logger.info("Upserted %d chunks into '%s'", len(chunks), self._collection)

    def search(
        self,
        query: str,
        top_k: int = settings.dense_top_k,
    ) -> list[RetrievedChunk]:
        """Dense cosine similarity search. Returns RetrievedChunks sorted by score."""
        query_vec = self._embedder.embed(query)
        response = self._client.query_points(
            collection_name=self._collection,
            query=query_vec,
            limit=top_k,
            with_payload=True,
        )
        return [
            RetrievedChunk(
                chunk=Chunk.from_payload(str(r.id), r.payload or {}),
                score=r.score,
                rank=i + 1,
            )
            for i, r in enumerate(response.points)
        ]

    def count(self) -> int:
        return self._client.count(collection_name=self._collection).count


@lru_cache(maxsize=1)
def get_store() -> VectorStore:
    return VectorStore()
