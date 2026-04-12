"""
Generate silver QA dataset for golden set creation.

Process:
  1. Pull representative chunks from each paper via Qdrant
  2. Use Claude Haiku to generate grounded QA pairs per paper
  3. Add cross-paper comparison questions using paper summaries
  4. Output to docs/golden_set_silver.json for human review

Question types:
  - factoid      : specific fact from one paper
  - methodology  : how something works / was designed
  - comparison   : contrast across 2 papers (cross-paper)
  - limitation   : what the paper says it can't do / future work
"""

import json
import os
import random
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from rich.console import Console
from rich.progress import track

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

console = Console()
OUT_PATH = Path("docs/golden_set_silver.json")

HAIKU = "claude-haiku-4-5-20251001"

SINGLE_PAPER_PROMPT = """\
You are building an evaluation dataset for a RAG system over academic AI papers.

Below are several text chunks from a paper titled: "{title}" ({year})

Your job: generate {n} diverse, high-quality question-answer pairs.

Rules:
- Each answer MUST be directly supported by the chunk text provided — no external knowledge
- Questions should be specific enough that only someone who read this paper could answer
- Include a mix of: factoid (specific numbers/names), methodology (how it works), limitation (what it can't do)
- For each pair, include the exact chunk_id that contains the answer
- Do NOT generate vague questions like "What is the main contribution?"
- Answers should be 1-3 sentences, factual and tight

Return ONLY a JSON array, no other text:
[
  {{
    "question": "...",
    "answer": "...",
    "chunk_id": "...",
    "type": "factoid|methodology|limitation",
    "paper_id": "{paper_id}"
  }}
]

Chunks:
{chunks}
"""

CROSS_PAPER_PROMPT = """\
You are building an evaluation dataset for a RAG system over academic AI papers.

Below are short summaries of papers in the corpus. Generate {n} cross-paper comparison questions.

Rules:
- Each question must require information from EXACTLY 2 different papers to answer fully
- Questions should contrast approaches, results, or design decisions between papers
- Answers must be grounded in the summaries provided — no hallucination
- Include the paper_ids of both papers involved
- Types: comparison (how do they differ?), synthesis (what do both agree on?)

Return ONLY a JSON array:
[
  {{
    "question": "...",
    "answer": "...",
    "chunk_ids": [],
    "type": "comparison|synthesis",
    "paper_ids": ["...", "..."]
  }}
]

Paper summaries:
{summaries}
"""


def get_diverse_chunks(qdrant: QdrantClient, paper_id: str, n: int = 6) -> list[dict]:
    """Pull n chunks from different sections of a paper."""
    result, _ = qdrant.scroll(
        collection_name="papers",
        scroll_filter={"must": [{"key": "paper_id", "match": {"value": paper_id}}]},
        limit=200,
        with_payload=True,
        with_vectors=False,
    )

    # Group by section, pick one chunk per section
    by_section: dict[str, list] = {}
    for pt in result:
        sec = pt.payload.get("section", "unknown")
        if sec not in by_section:
            by_section[sec] = []
        by_section[sec].append(pt)

    # Pick one chunk from each of n different sections
    selected = []
    sections = [s for s in by_section if s not in ("References", "Bibliography")]
    random.shuffle(sections)
    for sec in sections[:n]:
        chunk = random.choice(by_section[sec])
        p = chunk.payload
        if len(p["text"]) > 100:  # skip tiny chunks
            selected.append({
                "chunk_id": p["chunk_id"],
                "section": p["section"],
                "text": p["text"][:600],  # truncate for prompt
            })

    return selected[:n]


def generate_single_paper_qa(
    client: anthropic.Anthropic,
    paper_id: str,
    title: str,
    year: int,
    chunks: list[dict],
    n: int = 3,
) -> list[dict]:
    chunk_text = "\n\n".join(
        f"[chunk_id: {c['chunk_id']}] [{c['section']}]\n{c['text']}"
        for c in chunks
    )
    prompt = SINGLE_PAPER_PROMPT.format(
        title=title, year=year, paper_id=paper_id, n=n, chunks=chunk_text
    )
    resp = client.messages.create(
        model=HAIKU,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    # Extract JSON
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start == -1 or end == 0:
        return []
    try:
        pairs = json.loads(raw[start:end])
        # Validate
        valid = []
        for p in pairs:
            if all(k in p for k in ["question", "answer", "chunk_id", "type"]):
                p["paper_id"] = paper_id
                p["paper_title"] = title
                p["source"] = "single_paper"
                p["reviewed"] = False
                valid.append(p)
        return valid
    except json.JSONDecodeError:
        return []


def generate_cross_paper_qa(
    client: anthropic.Anthropic,
    paper_summaries: list[dict],
    n: int = 6,
) -> list[dict]:
    summaries_text = "\n\n".join(
        f"paper_id: {p['paper_id']}\ntitle: {p['title']} ({p['year']})\nsummary: {p['summary']}"
        for p in paper_summaries
    )
    prompt = CROSS_PAPER_PROMPT.format(n=n, summaries=summaries_text)
    resp = client.messages.create(
        model=HAIKU,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start == -1 or end == 0:
        return []
    try:
        pairs = json.loads(raw[start:end])
        valid = []
        for p in pairs:
            if all(k in p for k in ["question", "answer", "paper_ids"]):
                p.setdefault("chunk_ids", [])
                p["source"] = "cross_paper"
                p["reviewed"] = False
                valid.append(p)
        return valid
    except json.JSONDecodeError:
        return []


def get_paper_summary(qdrant: QdrantClient, paper_id: str, title: str) -> str:
    """Build a short summary from abstract/intro chunks."""
    result, _ = qdrant.scroll(
        collection_name="papers",
        scroll_filter={"must": [
            {"key": "paper_id", "match": {"value": paper_id}},
        ]},
        limit=200,
        with_payload=True,
        with_vectors=False,
    )
    # Find abstract or intro chunks
    for pt in result:
        sec = pt.payload.get("section", "").lower()
        if "abstract" in sec or "introduction" in sec:
            return pt.payload["text"][:500]
    # Fallback: first chunk
    if result:
        return result[0].payload["text"][:500]
    return title


def main():
    qdrant = QdrantClient(host="localhost", port=6333)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Get all papers
    paper_ids = set()
    offset = None
    paper_meta: dict[str, dict] = {}
    while True:
        result, next_offset = qdrant.scroll(
            collection_name="papers", limit=200, offset=offset,
            with_payload=True, with_vectors=False,
        )
        for pt in result:
            p = pt.payload
            pid = p["paper_id"]
            if pid not in paper_meta:
                paper_meta[pid] = {
                    "paper_id": pid,
                    "title": p["paper_title"],
                    "year": p["year"],
                }
        if next_offset is None:
            break
        offset = next_offset

    console.print(f"Found {len(paper_meta)} papers\n")

    all_qa: list[dict] = []

    # ── Single-paper questions ────────────────────────────────────────────────
    console.print("[bold]Generating single-paper QA pairs...[/bold]")
    for pid, meta in paper_meta.items():
        chunks = get_diverse_chunks(qdrant, pid, n=6)
        if not chunks:
            continue
        pairs = generate_single_paper_qa(
            client, pid, meta["title"], meta["year"], chunks, n=3
        )
        console.print(f"  {meta['title'][:50]:50s} → {len(pairs)} pairs")
        all_qa.extend(pairs)
        time.sleep(0.3)

    # ── Cross-paper questions ─────────────────────────────────────────────────
    console.print("\n[bold]Generating cross-paper QA pairs...[/bold]")
    summaries = []
    for pid, meta in paper_meta.items():
        summary = get_paper_summary(qdrant, pid, meta["title"])
        summaries.append({
            "paper_id": pid,
            "title": meta["title"],
            "year": meta["year"],
            "summary": summary,
        })

    cross_pairs = generate_cross_paper_qa(client, summaries, n=7)
    console.print(f"  Generated {len(cross_pairs)} cross-paper pairs")
    all_qa.extend(cross_pairs)

    # ── Save ──────────────────────────────────────────────────────────────────
    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(all_qa, f, indent=2)

    console.print(f"\n[bold green]Saved {len(all_qa)} pairs → {OUT_PATH}[/bold green]")
    console.print("\n[dim]Next: review each pair in docs/golden_set_silver.json[/dim]")
    console.print("[dim]Set 'reviewed': true and fix any wrong answers[/dim]")
    console.print("[dim]Delete pairs that are wrong or too vague[/dim]")

    # Preview
    console.print("\n[bold]Preview (first 3):[/bold]")
    for qa in all_qa[:3]:
        console.print(f"\n  Q: {qa['question']}")
        console.print(f"  A: {qa['answer'][:120]}")
        console.print(f"  Type: {qa['type']} | Source: {qa['source']}")


if __name__ == "__main__":
    main()
