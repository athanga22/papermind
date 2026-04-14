"""
Planner node for PaperMind (LangGraph).

Given a user query and the "universe" of ingested papers (titles), the planner
breaks the query into a list of specific retrieval search strings (sub-queries).
The output is a JSON list of strings, stored in `state["sub_queries"]`.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import List

import anthropic

from query.state import PaperMindState


DEFAULT_PARSED_DIR = Path("data/parsed")
# Planner uses Sonnet — query decomposition is reasoning-heavy work.
# Override with PAPERMIND_PLANNER_MODEL env var if needed.
PLANNER_MODEL = os.getenv("PAPERMIND_PLANNER_MODEL", "claude-haiku-4-5-20251001")


def _extract_title_from_markdown(markdown: str) -> str | None:
    """Return the first markdown header line as the paper title."""
    for line in markdown.split("\n"):
        if line.startswith("#"):
            return line.lstrip("#").strip() or None
    return None


def load_paper_titles(parsed_dir: Path = DEFAULT_PARSED_DIR, limit: int = 10) -> list[str]:
    """
    Load up to `limit` paper titles from LlamaParse markdown cache in `data/parsed/`.
    Title heuristic: first markdown header (`# ...`) in each file.
    """
    if not parsed_dir.exists():
        return []

    titles: list[str] = []
    for md_path in sorted(parsed_dir.glob("*.md")):
        try:
            text = md_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        title = _extract_title_from_markdown(text)
        if title:
            titles.append(title)
        if len(titles) >= limit:
            break
    return titles


_SYSTEM = """You are the PaperMind Research Planner. Your goal is to decompose research questions into a minimal, targeted search strategy for a corpus of 10 technical papers.

**YOUR MISSION:**
Transform a user question into the fewest surgical sub-queries needed to retrieve the answer. Retrieval is expensive — every extra sub-query adds 5 noisy chunks. Be precise, not exhaustive.

**QUERY COUNT RULES (hard limits — do not exceed):**
- Simple factoid (single fact, single paper): 2-3 sub-queries max
- Methodology / explanation (single paper): 3-4 sub-queries max
- Comparison or synthesis across 2 papers: 4-5 sub-queries max (2 per paper + 1 synthesis angle)
- Broad cross-paper synthesis (3+ papers): 5-6 sub-queries max

**SEARCH STRATEGY GUIDELINES:**
1. **Target the "How":** Search for methodologies and experimental setups, not just names.
2. **Isolate Variables:** For comparisons, one sub-query per paper being compared.
3. **Keyword Density:** Include specific technical terms the paper would use verbatim.
4. **Structural Anchors:** Use section-specific keywords to find data where it lives (e.g., "results table", "ablation study", "limitations section").
5. **Stop when you have enough:** If 2 sub-queries cover the question, use 2.

**CONSTRAINTS:**
- Output MUST be a valid JSON list of strings.
- Format: ["query 1", "query 2", "query 3"]
- No preamble, no markdown formatting, no explanations.
- Each query: 5-15 words.
- Maximum 6 sub-queries under any circumstances.
"""


def _parse_json_list(text: str) -> list[str]:
    """Parse a JSON list from model output (handles fenced/extra text)."""
    # Fast path: exact JSON
    try:
        data = json.loads(text)
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return [s.strip() for s in data if s.strip()]
    except Exception:
        pass

    # Try to extract the first JSON array substring.
    m = re.search(r"\[[\s\S]*\]", text)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        return []
    return []


def planner_node(state: PaperMindState) -> PaperMindState:
    """
    LangGraph node: plans `sub_queries` from `state["query"]`.

    Reads `max_sub_queries` set by the classifier node (Adaptive-RAG pattern)
    and enforces it as a hard cap in both the prompt and post-processing.
    """
    query = state["query"]
    # Hard cap set by classifier_node. Default 4 if classifier didn't run.
    max_sq = int(state.get("max_sub_queries") or 4)
    titles = load_paper_titles(limit=10)

    universe_context = (
        "Available papers (titles):\n" + "\n".join(f"- {t}" for t in titles)
        if titles
        else "Technical corpus."
    )

    user = (
        f"Universe: {universe_context}\n\n"
        f"User question: {query}\n\n"
        f"HARD LIMIT: Generate exactly {max_sq} sub-queries or fewer. "
        f"Do NOT exceed {max_sq}. "
        "Return ONLY a JSON list of surgical retrieval search strings."
    )

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=PLANNER_MODEL,
        max_tokens=400,
        system=_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )

    raw = resp.content[0].text
    sub_queries = _parse_json_list(raw)[:max_sq]  # hard truncate as backstop

    return {
        **state,
        "sub_queries": sub_queries,
    }

