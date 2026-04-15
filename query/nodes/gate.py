"""
Confidence gate for PaperMind (LangGraph).

Uses an LLM-as-judge sufficiency check (Self-RAG / CRAG pattern) instead of
a heuristic entity overlap score. The judge is asked one binary question:
"Does the retrieved context contain enough information to fully answer the question?"

Why LLM judge instead of the previous entity heuristic:
  - Entity mention coverage failed on cross-paper questions: entities from paper A
    don't appear in paper B's retrieval window, so the heuristic always scored < 0.6
    and all 27 cross-paper questions triggered max replans (90 chunks → noise).
  - Self-RAG (Asai et al., ICLR 2024) uses IsREL/IsSUP reflection tokens for this.
  - CRAG (Yan et al., 2024) uses a fine-tuned T5 evaluator with three categories:
    correct / ambiguous / incorrect. We use the simpler binary version.
  - MLflow's RetrievalSufficiency judge is the production reference implementation.

The judge uses Haiku (cheap, fast — ~0.2s, minimal cost). Hard replan cap is
enforced at MAX_REPLAN_ATTEMPTS regardless of judge output.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import anthropic
from langgraph.config import get_stream_writer

from query.state import PaperMindState

MAX_REPLAN_ATTEMPTS = int(os.getenv("PAPERMIND_MAX_REPLANS", "2"))
GATE_MODEL = os.getenv("PAPERMIND_GATE_MODEL", "claude-haiku-4-5-20251001")

_SUFFICIENCY_SYSTEM = """\
You are a retrieval quality judge for a RAG system.

Your ONLY job: detect when retrieval completely missed — i.e., ALL retrieved chunks \
are from clearly wrong papers or topics with zero connection to the question.

Return {"sufficient": false} ONLY when you are highly confident that NONE of the \
retrieved chunks contain any topic, method, system, or concept mentioned in the question.

Return {"sufficient": true} in ALL other cases, including when:
- Some chunks are relevant but others are not
- The context is partial or incomplete
- The context uses different terminology than the question
- You are unsure

The synthesis model handles partial context well. Only replan for clear retrieval failures.

Answer with ONLY valid JSON: {"sufficient": true|false}
"""


def _llm_sufficiency_judge(query: str, chunks: list[dict[str, Any]]) -> bool:
    """
    Ask Haiku to detect clear retrieval failures only.
    Returns False only when retrieval clearly missed entirely (all wrong papers).
    Defaults to True in all ambiguous cases.
    """
    if not chunks:
        return False

    # Pass only paper title + section per chunk — not chunk text.
    # The gate only needs to know *which papers* were retrieved, not their content.
    # Passing full text inflated gate latency to ~1.6s; title+section brings it to ~400ms.
    papers_seen = sorted({
        f"{c.get('paper_title', '?')} § {c.get('section', '?')}"
        for c in chunks
    })
    context_preview = "\n".join(f"  - {p}" for p in papers_seen)

    user = (
        f"Question: {query}\n\n"
        f"Papers/sections retrieved ({len(chunks)} chunks total):\n{context_preview}\n\n"
        "Did retrieval completely miss? (only say false if ALL entries are clearly wrong)"
    )

    try:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        resp = client.messages.create(
            model=GATE_MODEL,
            max_tokens=80,
            system=_SUFFICIENCY_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        raw = resp.content[0].text.strip()
        try:
            data = json.loads(raw)
        except Exception:
            m = re.search(r'\{.*?\}', raw, re.DOTALL)
            data = json.loads(m.group(0)) if m else {}
        return bool(data.get("sufficient", True))
    except Exception:
        # Fail open: if the judge errors, proceed to synthesis
        return True


def gate_node(state: PaperMindState) -> dict[str, Any]:
    """
    LangGraph node: call the LLM sufficiency judge and store confidence_score.

    confidence_score is 1.0 if sufficient, 0.0 if not — the gate_route function
    uses it for routing. The binary signal is cleaner than the old float heuristic.
    """
    get_stream_writer()({"type": "progress", "node": "gate", "message": "Checking retrieval quality..."})
    query = state.get("query") or ""
    chunks = list(state.get("retrieved_chunks") or [])

    sufficient = _llm_sufficiency_judge(query, chunks)
    return {"confidence_score": 1.0 if sufficient else 0.0}


def gate_route(state: PaperMindState) -> str:
    """
    LangGraph conditional router.

    Returns:
      - "synthesize" if context is sufficient OR replan budget exhausted
      - "replan" if context is insufficient AND budget remains

    Hard cap at MAX_REPLAN_ATTEMPTS enforces forward progress regardless of
    judge output (LangGraph PR #5954 pattern).
    """
    replan_count = int(state.get("replan_count") or 0)
    if replan_count >= MAX_REPLAN_ATTEMPTS:
        return "synthesize"

    score = float(state.get("confidence_score") or 0.0)
    return "replan" if score < 0.5 else "synthesize"
