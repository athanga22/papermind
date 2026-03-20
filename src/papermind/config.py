"""Central config — reads from .env or environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "papermind"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Chunking
    chunk_max_tokens: int = 400
    chunk_min_tokens: int = 100

    # Retrieval
    dense_top_k: int = 5


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
