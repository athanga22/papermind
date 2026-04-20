"""
Phase 6 — Semantic cache for PaperMind.

Uses GPTCache with:
  - OpenAI text-embedding-3-small for query embedding (same model as retrieval)
  - FAISS for vector similarity search
  - SQLite for metadata storage
  - SearchDistanceEvaluation for similarity scoring

When a new query arrives:
  1. Embed the query
  2. Search the cache for a semantically similar past query (cosine distance < threshold)
  3. If found: return cached answer (cache hit)
  4. If not: run the full agent pipeline, store the result, return it (cache miss)

The cache is persisted to disk at `data/cache/` so it survives restarts.

Usage:
    from query.cache import SemanticCache

    cache = SemanticCache()
    result = cache.get("What is SRAG?")
    if result is None:
        result = run_agent(query)
        cache.put("What is SRAG?", result)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

CACHE_DIR = Path(os.getenv("PAPERMIND_CACHE_DIR", "data/cache"))
CACHE_SIMILARITY_THRESHOLD = float(os.getenv("PAPERMIND_CACHE_THRESHOLD", "0.12"))
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536


def _make_embed_func(client: OpenAI):
    """Return an embedding function compatible with GPTCache (str -> np.array)."""
    def embed(text: str, **kwargs) -> np.ndarray:
        resp = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
        return np.array(resp.data[0].embedding, dtype=np.float32)
    return embed


class SemanticCache:
    """
    Semantic cache wrapping GPTCache with OpenAI embeddings + FAISS.

    Methods:
        get(query) -> dict | None   — returns cached agent state or None
        put(query, state) -> None   — stores agent state in cache
        stats() -> dict             — returns hit/miss counts
    """

    def __init__(
        self,
        cache_dir: Path | str = CACHE_DIR,
        similarity_threshold: float = CACHE_SIMILARITY_THRESHOLD,
    ):
        from gptcache import Cache
        from gptcache.manager import CacheBase, get_data_manager
        from gptcache.manager.vector_data import VectorBase
        from gptcache.processor.pre import get_prompt
        from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._threshold = similarity_threshold
        self._hits = 0
        self._misses = 0

        openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._embed_func = _make_embed_func(openai_client)

        cache_base = CacheBase(
            "sqlite",
            sql_url=f"sqlite:///{self._cache_dir / 'cache.db'}",
        )
        vector_base = VectorBase(
            "faiss",
            dimension=EMBEDDING_DIMENSION,
            index_path=str(self._cache_dir / "faiss.index"),
        )
        data_manager = get_data_manager(cache_base, vector_base)
        evaluation = SearchDistanceEvaluation()

        # GPTCache's put/get adapter API works on the global `cache` singleton.
        # We init the module-level global cache and also keep a reference.
        from gptcache.adapter.api import cache as global_cache
        self._cache = global_cache
        self._cache.init(
            pre_embedding_func=get_prompt,
            embedding_func=self._embed_func,
            data_manager=data_manager,
            similarity_evaluation=evaluation,
        )

    def get(self, query: str) -> dict | None:
        """
        Look up a semantically similar query in the cache.

        Returns the cached agent state dict if a match is found,
        otherwise None.
        """
        try:
            from gptcache.adapter.api import get as cache_get

            result = cache_get(query)
            if result is not None:
                state = json.loads(result)
                self._hits += 1
                logger.info("Cache HIT for: %s", query[:60])
                return state
        except Exception as e:
            logger.debug("Cache lookup error (treated as miss): %s", e)

        self._misses += 1
        return None

    def put(self, query: str, state: dict) -> None:
        """
        Store an agent result in the cache.

        The state dict is JSON-serialized and stored alongside the query embedding.
        """
        try:
            from gptcache.adapter.api import put as cache_put

            cache_put(query, json.dumps(state, default=str))
            logger.info("Cache PUT for: %s", query[:60])
        except Exception as e:
            logger.warning("Cache put error (non-fatal): %s", e)

    def stats(self) -> dict:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(1, self._hits + self._misses),
        }

    def flush(self) -> None:
        """Persist any pending writes to disk."""
        try:
            self._cache.flush()
        except Exception:
            pass
