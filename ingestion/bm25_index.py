"""
Step 5 — BM25S keyword index (persisted to disk).

Strategy:
  - Uses bm25s library (fast, no external dependencies)
  - Index is built from all chunk texts + stored with chunk_id mapping
  - Persisted to data/bm25/ — rebuilt if chunks change
  - Corpus stored alongside index for retrieval (returns chunk_ids)
  - Idempotent: build() overwrites existing index

The BM25 index is used at query time alongside dense retrieval (trimodal).
"""

import json
from pathlib import Path

import bm25s

from ingestion.models import Chunk

BM25_DIR = Path("data/bm25")


class BM25Index:
    """
    Manages the BM25S keyword index over all paper chunks.
    """

    def __init__(self, index_dir: Path = BM25_DIR) -> None:
        self._dir = index_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._mapping_path = self._dir / "chunk_ids.json"

        self._model: bm25s.BM25 | None = None
        self._chunk_ids: list[str] = []

    def build(self, chunks: list[Chunk], show_progress: bool = False) -> None:
        """
        Build (or rebuild) the BM25 index from the given chunks.
        Persists the index and chunk_id mapping to disk.
        """
        texts = [c.text for c in chunks]
        self._chunk_ids = [c.chunk_id for c in chunks]

        corpus_tokens = bm25s.tokenize(
            texts,
            stopwords="english",
            show_progress=show_progress,
        )

        self._model = bm25s.BM25()
        self._model.index(corpus_tokens, show_progress=show_progress)

        # Save BM25 index (without corpus — we store chunk_ids separately)
        self._model.save(str(self._dir), corpus=None)

        # Save chunk_id mapping (position → chunk_id)
        with open(self._mapping_path, "w") as f:
            json.dump(self._chunk_ids, f)

    def load(self) -> None:
        """Load a previously saved index from disk."""
        if not self._mapping_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {self._dir}. Run build() first."
            )
        self._model = bm25s.BM25.load(str(self._dir), load_corpus=False)
        with open(self._mapping_path) as f:
            self._chunk_ids = json.load(f)

    def query(self, text: str, k: int = 10) -> list[tuple[str, float]]:
        """
        Query the index. Returns list of (chunk_id, score) sorted by score desc.
        """
        if self._model is None:
            raise RuntimeError("Index not loaded. Call build() or load() first.")

        query_tokens = bm25s.tokenize(
            [text],
            stopwords="english",
            show_progress=False,
        )
        results, scores = self._model.retrieve(
            query_tokens,
            k=min(k, len(self._chunk_ids)),
            show_progress=False,
            return_as="tuple",
        )
        # results shape: (n_queries, k) — indices into corpus
        # scores shape: (n_queries, k)
        hits = []
        for idx, score in zip(results[0], scores[0]):
            chunk_id = self._chunk_ids[int(idx)]
            hits.append((chunk_id, float(score)))
        return hits

    @property
    def size(self) -> int:
        return len(self._chunk_ids)

    def exists(self) -> bool:
        return self._mapping_path.exists()
