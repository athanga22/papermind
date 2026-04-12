"""
Step 4 — Dense embedding + Qdrant storage.

Strategy:
  - Model: text-embedding-3-small (1536 dims), cosine similarity
  - Batch size: 100 (OpenAI limit: 2048 inputs, but 100 keeps latency low)
  - Qdrant collection: "papers" — created idempotently on first run
  - Point IDs: stable UUID5 derived from chunk_id (reproducible re-runs)
  - Payload stored per point: all Chunk fields except the embedding itself
  - Idempotent: upsert (not insert) so re-running is safe

No LLM calls. Pure embedding → vector store.
"""

import os
import uuid
from typing import Iterator

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ingestion.models import Chunk

# ── Constants ─────────────────────────────────────────────────────────────────

COLLECTION_NAME = "papers"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
EMBED_BATCH_SIZE = 100

# Stable UUID namespace for chunk_id → point UUID derivation
_UUID_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # uuid.NAMESPACE_URL


def _chunk_to_point_id(chunk_id: str) -> str:
    """Derive a stable UUID from chunk_id for Qdrant point identity."""
    return str(uuid.uuid5(_UUID_NS, chunk_id))


def _chunk_payload(chunk: Chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "paper_id": chunk.paper_id,
        "paper_title": chunk.paper_title,
        "authors": chunk.authors,
        "year": chunk.year,
        "section": chunk.section,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "is_table": chunk.is_table,
        "contains_math": chunk.contains_math,
    }


def _batched(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ── Embedder ──────────────────────────────────────────────────────────────────

class ChunkEmbedder:
    """
    Embeds chunks with text-embedding-3-small and upserts them into Qdrant.
    One instance per pipeline run — reuse across papers.
    """

    def __init__(self) -> None:
        self._openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._qdrant = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
        )
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it doesn't exist."""
        existing = {c.name for c in self._qdrant.get_collections().collections}
        if COLLECTION_NAME not in existing:
            self._qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qmodels.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=qmodels.Distance.COSINE,
                ),
            )

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Call OpenAI embeddings API for a batch of texts."""
        response = self._openai.embeddings.create(
            input=texts,
            model=EMBEDDING_MODEL,
        )
        # Results are ordered by index
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

    def embed_and_store(self, chunks: list[Chunk]) -> int:
        """
        Embed all chunks and upsert into Qdrant.
        Returns the number of points upserted.
        """
        total = 0
        for batch in _batched(chunks, EMBED_BATCH_SIZE):
            texts = [c.text for c in batch]
            embeddings = self._embed_texts(texts)

            points = [
                qmodels.PointStruct(
                    id=_chunk_to_point_id(chunk.chunk_id),
                    vector=embedding,
                    payload=_chunk_payload(chunk),
                )
                for chunk, embedding in zip(batch, embeddings)
            ]
            self._qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True,
            )
            total += len(points)

        return total

    def collection_count(self) -> int:
        """Return current point count in the collection."""
        info = self._qdrant.get_collection(COLLECTION_NAME)
        return info.points_count
