"""Central configuration via pydantic-settings — reads from .env or environment."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Qdrant ──────────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "papermind"

    # ── Embeddings ───────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # ── Reranker ─────────────────────────────────────────────────────────
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # If top reranked chunk scores below this, trigger query rewriting
    reranker_threshold: float = 0.3

    # ── Retrieval ────────────────────────────────────────────────────────
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rerank_top_k: int = 5
    rrf_k: int = 60  # RRF constant — higher = smoother rank fusion

    # ── Chunking ─────────────────────────────────────────────────────────
    chunk_max_tokens: int = 400
    chunk_min_tokens: int = 100

    # ── Claude ───────────────────────────────────────────────────────────
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    claude_model: str = "claude-opus-4-6"
    claude_max_tokens: int = 2048
    claude_temperature: float = 0.0

    # ── LangSmith ────────────────────────────────────────────────────────
    langsmith_api_key: str = ""
    langchain_tracing_v2: bool = False
    langchain_project: str = "papermind"

    # ── RAGAS (uses OpenAI for evaluation LLM by default) ────────────────
    openai_api_key: str = ""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Module-level singleton for convenience
settings = get_settings()
