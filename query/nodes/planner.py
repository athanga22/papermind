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


_SYSTEM = """You are the PaperMind Research Planner. Your goal is to decompose complex research inquiries into a multi-step search strategy for a corpus of 10 technical papers.

**YOUR MISSION:**
Transform a single user question into surgical sub-queries that will be executed across a hybrid (Vector + BM25) retrieval system. Your search plan must enable cross-document synthesis.

**SEARCH STRATEGY GUIDELINES:**
1. **Target the "How":** Don't just search for names; search for methodologies, algorithms, and experimental setups (e.g., "implementation details of [Method]", "statistical significance of [Result]").
2. **Isolate Variables:** If comparing two things, create separate queries for each to avoid "query muddying."
3. **Keyword Density:** Include specific technical terms, units of measure, or standard nomenclature that a researcher would use (e.g., "p-value," "latency in ms," "O(n) complexity").
4. **Structural Anchors:** Target specific document sections. If you know a claim likely lives in a certain paper, include the paper title in the sub-query (e.g., "Paper xRouter evaluation section results table"). Also use section-specific keywords to find data where it lives (e.g., "Ablation study," "Related works," "Table 1 results," "Proposed framework").
5. **Conflict Discovery:** If the query implies a tension, search for the specific limitations or boundary conditions of each paper's claims.

**CONSTRAINTS:**
- Output MUST be a valid JSON list of strings. 
- Format: ["query 1", "query 2", "query 3"]
- No preamble, no markdown formatting, no explanations.
- Each query should be 5-12 words.
- Use distinct angles for each query to maximize context recall.
- Diversity Constraint: Each sub-query in the JSON list must target a distinctly different semantic aspect of the query. Do not repeat terms across sub-queries unless they are the primary subject.
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

    Reads paper titles from `data/parsed/` and includes them in the prompt so the
    model knows the corpus universe.
    """
    query = state["query"]
    titles = load_paper_titles(limit=10)

    # If no titles, prompt more broadly. If titles exist, be surgical.
    universe_context = (
        "Available papers (titles):\n" + "\n".join(f"- {t}" for t in titles)
        if titles
        else "Technical corpus."
    )

    user = (
        f"Universe: {universe_context}\n\n"
        f"User question: {query}\n\n"
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
    sub_queries = _parse_json_list(raw)

    return {
        **state,
        "sub_queries": sub_queries,
    }

