"""
PaperMind FastAPI server.

Two endpoints:
  POST /query         — blocking JSON response (for scripts / eval)
  POST /query/stream  — Server-Sent Events stream (for the UI)

SSE stream event shape:
  {"type": "progress", "node": "<name>", "message": "<human label>"}
  {"type": "token",    "content": "<text chunk>"}
  {"type": "done",     "answer": "<full answer>", "confidence": "<high|medium|low>"}
  {"type": "error",    "detail": "<message>"}

Run:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from query.agent import app as graph, initial_state_for  # noqa: E402  (after dotenv)


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="PaperMind", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    use_cache: bool = True


class QueryResponse(BaseModel):
    answer: str
    confidence: str
    sub_queries: list[str]
    n_chunks: int


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_initial_state(question: str):
    return initial_state_for(question)


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Blocking endpoint — returns full answer as JSON."""
    from query.agent import run_agent
    state = run_agent(req.question, use_cache=req.use_cache)
    return QueryResponse(
        answer=state.get("synthesis") or "",
        confidence=(
            "high"   if (state.get("confidence_score") or 0) >= 0.8 else
            "medium" if (state.get("confidence_score") or 0) >= 0.5 else
            "low"
        ),
        sub_queries=list(state.get("sub_queries") or []),
        n_chunks=len(state.get("retrieved_chunks") or []),
    )


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """
    Streaming endpoint — Server-Sent Events.

    Progress events fire as each LangGraph node starts. Synthesis tokens stream
    word-by-word via get_stream_writer() in synthesis_node. The final `done`
    event carries the complete assembled answer.

    Timeline for a typical cross-paper comparison:
      t+0s  → progress: classifying
      t+1s  → progress: planning
      t+2s  → progress: retrieving (fires once per sub-query, in parallel)
      t+3s  → progress: ranking
      t+3s  → progress: gate check
      t+4s  → progress: synthesizing
      t+4s  → tokens start streaming
      t+9s  → done
    """
    async def event_gen() -> AsyncIterator[str]:
        initial = _build_initial_state(req.question)
        full_answer_parts: list[str] = []
        final_state: dict = {}

        try:
            async for chunk in graph.astream(
                initial,
                stream_mode=["custom", "updates"],
            ):
                # LangGraph yields (mode, payload) tuples in multi-mode streaming
                mode, payload = chunk if isinstance(chunk, tuple) else ("updates", chunk)

                if mode == "custom":
                    event = payload  # dict emitted by get_stream_writer()
                    if event.get("type") == "token":
                        full_answer_parts.append(event["content"])
                    yield _sse(event)

                elif mode == "updates":
                    # Capture final state fields from node updates
                    for node_name, node_update in payload.items():
                        if isinstance(node_update, dict):
                            final_state.update(node_update)

            # Assemble done event
            answer = "".join(full_answer_parts) or final_state.get("synthesis", "")
            confidence_score = float(final_state.get("confidence_score") or 0.0)
            confidence = (
                "high"   if confidence_score >= 0.8 else
                "medium" if confidence_score >= 0.5 else
                "low"
            )
            yield _sse({"type": "done", "answer": answer, "confidence": confidence})

        except Exception as exc:
            yield _sse({"type": "error", "detail": str(exc)})

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":   "no-cache",
            "X-Accel-Buffering": "no",   # critical for nginx — disables proxy buffering
        },
    )


@app.get("/health")
def health():
    return {"status": "ok"}
