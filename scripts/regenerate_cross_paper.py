"""
Regenerate the 7 cross-paper questions in golden_set_silver.json.

Problems with the current questions:
  1. Question text says "paper f508af927f8b" instead of the real title
  2. Answers say "Paper 4dd0dd178095" instead of the paper name
  3. LLM-generated answers may not reflect actual paper content

Fix: for each paper pair, pull the top-relevant chunks from both papers,
feed them to Claude Haiku, and generate a grounded cross-paper question+answer
that uses real titles and is factually anchored in the actual text.
"""

import json
import os
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv(Path(__file__).parent.parent / ".env", override=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

from query.retriever import TrimodalRetriever

# ── Paper title map ───────────────────────────────────────────────────────────

PAPER_TITLES = {
    "f508af927f8b": "Memory in the LLM Era: Modular Architectures and Strategies in a Unified Framework",
    "210e11e8f71b": "Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG",
    "4dd0dd178095": "xRouter: Training Cost-Aware LLM Orchestration via Reinforcement Learning",
    "19c5e058258d": "Universal Model Routing for Efficient LLM Inference",
    "7efed776d7d0": "Reasoning RAG via System 1 or System 2: A Survey on Reasoning Agentic RAG",
    "97496e256274": "From BM25 to Corrective RAG: Benchmarking Retrieval Strategies for Text-and-Table Documents",
    "ed4d0ac35253": "SRAG: RAG with Structured Data Improves Vector Retrieval",
    "7259de5942f0": "ClinicalAgents: Multi-Agent Orchestration for Clinical Decision Making with Dual-Memory",
    "492b1abd5824": "A-RAG: Scaling Agentic RAG via Hierarchical Retrieval Interfaces",
    "5194931d0bc1": "BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute",
}

# ── Cross-paper pairs + topic hints ──────────────────────────────────────────
# Each entry: (paper_id_A, paper_id_B, question_type, search_query)
# search_query drives what we pull from each paper to ground the question

CROSS_PAIRS = [
    (
        "f508af927f8b", "210e11e8f71b",
        "comparison",
        "modular memory architecture LLM agents",
        "How does {A} conceptualise memory as a modular component for LLM agents, "
        "and how does {B} incorporate memory into its agentic RAG taxonomy?",
    ),
    (
        "4dd0dd178095", "19c5e058258d",
        "comparison",
        "LLM routing cost performance tradeoff architecture",
        "What are the core architectural differences between how {A} and {B} approach "
        "the problem of routing queries to different LLMs?",
    ),
    (
        "7efed776d7d0", "210e11e8f71b",
        "synthesis",
        "evolution from static RAG to agentic reasoning pipeline",
        "How do {A} and {B} both frame the progression from static RAG to agentic "
        "reasoning, and where do their characterisations differ?",
    ),
    (
        "97496e256274", "ed4d0ac35253",
        "comparison",
        "retrieval improvement hybrid BM25 structured data",
        "How do {A} and {B} differ in their approaches to improving retrieval "
        "effectiveness beyond standard dense vector search?",
    ),
    (
        "7259de5942f0", "492b1abd5824",
        "synthesis",
        "multi-agent system architecture memory hierarchical retrieval",
        "What shared problem with rigid single-agent systems do {A} and {B} each "
        "solve, and what different architectural strategies do they use?",
    ),
    (
        "5194931d0bc1", "4dd0dd178095",
        "comparison",
        "cost performance adaptive routing LLM inference",
        "Both {A} and {B} tackle the cost-quality tradeoff in LLM deployment — "
        "how do their routing strategies differ in design and training?",
    ),
    (
        "f508af927f8b", "7259de5942f0",
        "synthesis",
        "memory management storage retrieval agent",
        "What complementary perspectives on agent memory do {A} and {B} provide, "
        "and how do their memory management strategies differ in scope?",
    ),
]

# ── Haiku prompt ──────────────────────────────────────────────────────────────

_SYSTEM = """\
You are an expert research assistant generating evaluation questions for a RAG system.
You will receive text chunks from TWO academic papers and a suggested question framing.

Your task: write ONE cross-paper comparison or synthesis question and a concise answer
(3-5 sentences) that is STRICTLY grounded in the provided chunks.

Rules:
- Use the full paper titles, never paper IDs
- The answer must be directly supported by the chunk text — no extrapolation
- Question should require reading BOTH papers to answer fully
- Answer should name which paper claims what, using short title references
- Keep the answer factual and specific, not vague or generic
- Return ONLY valid JSON: {"question": "...", "answer": "..."}
"""


def pull_chunks(retriever: TrimodalRetriever, paper_id: str, query: str, n: int = 4) -> list[str]:
    """Pull top-n chunks from a specific paper using semantic search."""
    hits = retriever.retrieve(query, top_k=20, use_dense=True, use_bm25=True, use_graph=False)
    paper_hits = [h for h in hits if h.paper_id == paper_id]
    # If retrieval didn't find enough from this paper, fall back to scroll
    if len(paper_hits) < 2:
        from qdrant_client import QdrantClient
        qc = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"),
                          port=int(os.getenv("QDRANT_PORT", "6333")))
        results, _ = qc.scroll(
            collection_name="papers",
            scroll_filter={"must": [{"key": "paper_id", "match": {"value": paper_id}}]},
            limit=n, with_payload=True, with_vectors=False,
        )
        return [r.payload["text"] for r in results]
    return [h.text for h in paper_hits[:n]]


def generate_qa(client: anthropic.Anthropic, title_a: str, title_b: str,
                chunks_a: list[str], chunks_b: list[str], hint: str) -> dict:
    chunks_block = ""
    chunks_block += f"\n\n=== Chunks from: {title_a} ===\n"
    for i, c in enumerate(chunks_a, 1):
        chunks_block += f"\n[Chunk A{i}]\n{c}\n"
    chunks_block += f"\n\n=== Chunks from: {title_b} ===\n"
    for i, c in enumerate(chunks_b, 1):
        chunks_block += f"\n[Chunk B{i}]\n{c}\n"

    user_msg = (
        f"Question framing hint: {hint}\n\n"
        f"Paper A title: {title_a}\n"
        f"Paper B title: {title_b}\n"
        f"{chunks_block}\n\n"
        "Generate the JSON question+answer now."
    )

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = resp.content[0].text.strip()
    # Extract JSON
    import re
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON in response: {raw}")
    return json.loads(m.group())


def main():
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    with open("docs/golden_set_silver.json") as f:
        golden = json.load(f)

    # Keep single-paper questions, replace cross-paper ones
    single = [q for q in golden if q["source"] == "single_paper"]
    new_cross = []

    print(f"Regenerating {len(CROSS_PAIRS)} cross-paper questions...\n")

    with TrimodalRetriever() as retriever:
        for i, (pid_a, pid_b, qtype, query, hint) in enumerate(CROSS_PAIRS, 1):
            title_a = PAPER_TITLES[pid_a]
            title_b = PAPER_TITLES[pid_b]
            hint_filled = hint.format(A=title_a.split(":")[0], B=title_b.split(":")[0])

            print(f"[{i}/7] {title_a.split(':')[0]} × {title_b.split(':')[0]}")

            chunks_a = pull_chunks(retriever, pid_a, query, n=4)
            chunks_b = pull_chunks(retriever, pid_b, query, n=4)

            print(f"       chunks pulled: A={len(chunks_a)}, B={len(chunks_b)}")

            try:
                qa = generate_qa(client, title_a, title_b, chunks_a, chunks_b, hint_filled)
                print(f"       Q: {qa['question'][:80]}...")
                print(f"       A: {qa['answer'][:80]}...")
            except Exception as e:
                print(f"       ERROR: {e}")
                qa = {"question": hint_filled, "answer": "Could not generate answer."}

            new_cross.append({
                "question": qa["question"],
                "answer": qa["answer"],
                "chunk_id": None,
                "type": qtype,
                "paper_ids": [pid_a, pid_b],
                "paper_titles": [title_a, title_b],
                "source": "cross_paper",
                "reviewed": False,
            })

            time.sleep(0.3)
            print()

    # Rebuild golden set: single first, then regenerated cross
    updated = single + new_cross

    with open("docs/golden_set_silver.json", "w") as f:
        json.dump(updated, f, indent=2)

    print(f"Saved {len(updated)} questions ({len(single)} single + {len(new_cross)} cross-paper)")


if __name__ == "__main__":
    main()
