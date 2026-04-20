"""
Eval set helper — find which chunk(s) contain the answer to a question.

Usage:
    python scripts/find_chunk.py "How does BEST-Route estimate output token length?"

Prints top-5 retrieved chunks with chunk_id, paper, section, score, and text
so you can verify which one(s) actually contain your reference answer.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from query.retriever import TrimodalRetriever  # noqa: E402


def find_chunk(query: str, top_k: int = 5) -> None:
    with TrimodalRetriever() as r:
        results = r.retrieve(query, top_k=top_k)

    if not results:
        print("No results found.")
        return

    print(f"\nTop {len(results)} chunks for: \"{query}\"\n")
    print("=" * 80)

    for rank, chunk in enumerate(results, 1):
        print(f"\n[{rank}] chunk_id : {chunk.chunk_id}")
        print(f"     score    : {chunk.score:.4f}  sources: {chunk.sources}")
        print(f"     paper    : {chunk.paper_title}")
        print(f"     section  : {chunk.section}")
        print(f"     text     : {chunk.text[:400]}...")
        print("-" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/find_chunk.py \"your question here\"")
        sys.exit(1)
    find_chunk(" ".join(sys.argv[1:]))
