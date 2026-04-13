"""
Langfuse observability — SDK v4 client singleton.

Connects to Langfuse Cloud (us.cloud.langfuse.com) using keys from .env.
Instruments the RAG pipeline via context-manager spans in pipeline.py.

Env vars read:
  LANGFUSE_PUBLIC_KEY   — project public key from Langfuse Cloud dashboard
  LANGFUSE_SECRET_KEY   — project secret key
  LANGFUSE_BASE_URL     — defaults to https://us.cloud.langfuse.com

Graceful degradation:
  If keys are missing or auth fails, get_client() returns None and all
  tracing calls in pipeline.py short-circuit silently. Pipeline keeps running.

Usage (in pipeline.py):
    from query.tracing import get_client

    lf = get_client()
    if lf:
        with lf.start_as_current_observation(name="rag-query", as_type="agent",
                                              input={"query": query}):
            ...
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

logger = logging.getLogger(__name__)

_client = None          # langfuse.Langfuse | None
_init_done = False      # only try once per process


def get_client():
    """
    Return the process-wide Langfuse client, or None if unavailable.

    Initialises on first call. Safe to call repeatedly — returns the same instance.
    Returns None (not an exception) when tracing is disabled or misconfigured.
    """
    global _client, _init_done
    if _init_done:
        return _client
    _init_done = True

    if os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower() == "false":
        logger.debug("Langfuse tracing disabled via LANGFUSE_TRACING_ENABLED=false")
        return None

    pk   = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    sk   = os.getenv("LANGFUSE_SECRET_KEY", "")
    host = os.getenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")

    if not pk or not sk:
        logger.warning(
            "Langfuse keys not set — tracing disabled. "
            "Set LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY in .env"
        )
        return None

    try:
        from langfuse import Langfuse
        lf = Langfuse(public_key=pk, secret_key=sk, host=host)
        if lf.auth_check():
            logger.info("Langfuse tracing enabled → %s", host)
            _client = lf
        else:
            logger.warning("Langfuse auth failed — tracing disabled")
    except Exception as exc:
        logger.warning("Langfuse init error: %s — tracing disabled", exc)

    return _client


def shutdown() -> None:
    """Flush all pending spans. Call at process exit."""
    if _client is not None:
        try:
            _client.flush()
        except Exception:
            pass
