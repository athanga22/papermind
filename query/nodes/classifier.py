"""
Query classifier node — Adaptive-RAG pattern (Jeong et al., NAACL 2024).

Routes incoming queries into three complexity tiers before the planner runs:
  - simple   → 1-2 sub-queries (single-paper factoids, direct lookups)
  - moderate → 3-4 sub-queries (methodology explanations, single-paper analysis)
  - complex  → 5-6 sub-queries (cross-paper comparisons, synthesis questions)

Why a separate classifier node instead of prompt instructions in the planner:
  The Adaptive-RAG paper showed prompt constraints alone are unreliable — the LLM
  ignores soft limits when it "feels" the question is hard. Routing to different
  strategies before planning is the correct fix. RAGRouter-Bench (2604.03455) found
  a classifier achieves 93.2% accuracy with 28% token savings vs always using
  expensive multi-hop retrieval.

We use Haiku (cheap, fast) with structured output. The planner reads
`state["max_sub_queries"]` and treats it as a hard cap.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import anthropic

from query.state import PaperMindState

CLASSIFIER_MODEL = os.getenv("PAPERMIND_CLASSIFIER_MODEL", "claude-haiku-4-5-20251001")

_SYSTEM = """You are a query complexity classifier for a research RAG system.

Classify the user question into one of three tiers:

**simple** — Single-paper factoid: one specific fact, figure, or definition.
  Examples: "What is X?", "What value does Y report?", "What are the three Z?"
  Sub-query budget: 2

**moderate** — Single-paper methodology/limitation/explanation requiring context assembly.
  Examples: "How does X work?", "Why does Y fail on Z?", "What are the key components of X?"
  Sub-query budget: 4

**complex** — Cross-paper comparison, synthesis, or multi-concept reasoning.
  Examples: "Compare X and Y", "How do papers A and B differ on Z?", "What shared pattern..."
  Sub-query budget: 6

Respond with ONLY valid JSON:
{"complexity": "simple"|"moderate"|"complex", "max_sub_queries": 2|4|6}
No preamble, no explanation."""


def classifier_node(state: PaperMindState) -> dict[str, Any]:
    """
    LangGraph node: classify query complexity and set sub-query budget.

    Writes `max_sub_queries` into state. The planner node reads this
    and treats it as a hard cap.
    """
    query = state.get("query") or ""

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=CLASSIFIER_MODEL,
        max_tokens=60,
        system=_SYSTEM,
        messages=[{"role": "user", "content": query}],
    )

    raw = resp.content[0].text.strip()
    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r'\{.*?\}', raw, re.DOTALL)
        try:
            data = json.loads(m.group(0)) if m else {}
        except Exception:
            data = {}

    max_sq = int(data.get("max_sub_queries", 4))
    # Clamp to valid values
    max_sq = max(2, min(6, max_sq))

    return {"max_sub_queries": max_sq}
