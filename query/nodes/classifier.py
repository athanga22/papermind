"""
Query classifier node — Adaptive-RAG pattern (Jeong et al., NAACL 2024).

Routes incoming queries into four complexity tiers before the planner runs:
  - simple     → 2 sub-queries  (single-paper factoids, direct lookups)
  - moderate   → 4 sub-queries  (methodology explanations, single-paper analysis)
  - comparison → 3 sub-queries  (exactly-2-paper comparisons)
  - synthesis  → 5 sub-queries  (3+ paper cross-paper synthesis)

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
from langgraph.config import get_stream_writer

from query.nodes.planner import load_paper_titles
from query.state import PaperMindState

CLASSIFIER_MODEL = os.getenv("PAPERMIND_CLASSIFIER_MODEL", "claude-haiku-4-5-20251001")

_SYSTEM_TEMPLATE = """You are a query complexity classifier for a research RAG system.

Classify the user question into one of four tiers:

**simple** — Single-paper factoid: one specific fact, figure, or definition.
  Examples: "What is X?", "What value does Y report?", "What are the three Z?"
  Sub-query budget: 2

**moderate** — Single-paper methodology/limitation/explanation requiring context assembly.
  Examples: "How does X work?", "Why does Y fail on Z?", "What are the key components of X?"
  Sub-query budget: 4

**comparison** — Comparison between exactly 2 papers on a specific aspect.
  Examples: "Compare X and Y", "How do papers A and B differ on Z?", "Which approach is better?"
  Sub-query budget: 4

**synthesis** — Cross-paper reasoning spanning 3 or more papers, or broad thematic questions.
  Examples: "What shared pattern...", "How do all papers address X?", "Summarize the field on Y"
  Sub-query budget: 5

For **comparison** and **synthesis** ONLY: extract the paper titles the question
explicitly names. Match against the corpus titles below — use the exact corpus
title string, not the user's paraphrase. If the question names no papers (pure
thematic question), return an empty list.

Available papers in the corpus:
{titles_block}

Respond with ONLY valid JSON:
{{"complexity": "simple"|"moderate"|"comparison"|"synthesis",
 "max_sub_queries": 2|4|4|5,
 "target_papers": ["<corpus title 1>", "<corpus title 2>", ...]}}

No preamble, no explanation."""


def classifier_node(state: PaperMindState) -> dict[str, Any]:
    """
    LangGraph node: classify query complexity and set sub-query budget.

    Writes:
      - `max_sub_queries`: hard cap read by the planner (2/3/4/5)
      - `target_papers`: corpus titles referenced by the question (used by the
        planner to generate paper-scoped sub-queries for comparison/synthesis
        questions, so retrieval covers both papers instead of whichever has
        denser keyword overlap)
    """
    get_stream_writer()({"type": "progress", "node": "classifier", "message": "Classifying query..."})
    query = state.get("query") or ""

    titles = load_paper_titles(limit=10)
    titles_block = "\n".join(f"- {t}" for t in titles) if titles else "(none loaded)"
    system_prompt = _SYSTEM_TEMPLATE.format(titles_block=titles_block)

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=CLASSIFIER_MODEL,
        max_tokens=300,
        system=system_prompt,
        messages=[{"role": "user", "content": query}],
    )

    raw = resp.content[0].text.strip()
    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        try:
            data = json.loads(m.group(0)) if m else {}
        except Exception:
            data = {}

    max_sq = int(data.get("max_sub_queries", 4))
    # Clamp to valid range: simple=2, comparison=3, moderate=4, synthesis=5
    max_sq = max(2, min(5, max_sq))  # valid range: 2/4/4/5

    # Validate target_papers against the corpus — drop anything the classifier
    # hallucinated. Keep only exact corpus-title matches.
    raw_targets = data.get("target_papers") or []
    if not isinstance(raw_targets, list):
        raw_targets = []
    corpus_set = set(titles)
    target_papers = [t for t in raw_targets if isinstance(t, str) and t in corpus_set]

    return {"max_sub_queries": max_sq, "target_papers": target_papers}
