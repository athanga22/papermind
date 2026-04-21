"""
Eval set helper — find and inspect chunks.

Usage:
    # Search by question — shows top-k chunks
    python scripts/find_chunk.py "How does BEST-Route estimate output token length?"

    # Inspect a specific chunk by ID — shows full text
    python scripts/find_chunk.py --id 4dd0dd178095_0025
"""

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

from query.retriever import TrimodalRetriever


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
        print(f"     text     :\n{chunk.text[:600]}...")
        print("-" * 80)


def inspect_chunk(chunk_id: str) -> None:
    with TrimodalRetriever() as r:
        r._fetch_payloads([chunk_id])
        payload = r._payload_cache.get(chunk_id)

    if not payload:
        print(f"Chunk not found: {chunk_id}")
        return

    print(f"\nchunk_id : {chunk_id}")
    print(f"paper_id : {payload.get('paper_id')}")
    print(f"paper    : {payload.get('paper_title')}")
    print(f"section  : {payload.get('section')}")
    print(f"\n{'='*80}\n")
    print(payload.get("text", ""))
    print(f"\n{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("query",   nargs="?", help="Question to search")
    group.add_argument("--id",    dest="chunk_id", help="Chunk ID to inspect fully")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if args.chunk_id:
        inspect_chunk(args.chunk_id)
    else:
        find_chunk(args.query, top_k=args.top_k)
