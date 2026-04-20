"""
CLI runner for the PaperMind LangGraph agentic pipeline.

Usage:
    python -m scripts.run_agent "Your question here"
    python -m scripts.run_agent "Your question here" --verbose

Flags:
    --verbose   Print retrieved chunks in addition to the answer.
    --session   Optional session ID for Langfuse grouping (defaults to a UUID).
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from query.agent import run_agent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PaperMind agentic RAG — LangGraph pipeline"
    )
    parser.add_argument("query", nargs="?", help="Question to answer")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print retrieved chunks"
    )
    parser.add_argument(
        "--session", default=None, help="Langfuse session ID (optional)"
    )
    args = parser.parse_args()

    query = args.query
    if not query:
        # Interactive mode
        try:
            query = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            sys.exit(0)

    if not query:
        print("No question provided.", file=sys.stderr)
        sys.exit(1)

    session_id = args.session or str(uuid.uuid4())

    print(f"\n[PaperMind] Query: {query}")
    print(f"[PaperMind] Session: {session_id}\n")

    state = run_agent(query, session_id=session_id)

    sub_queries = state.get("sub_queries") or []
    replan_count = state.get("replan_count") or 0
    confidence = state.get("confidence_score") or 0.0
    chunks = state.get("retrieved_chunks") or []
    synthesis = state.get("synthesis") or "(no answer)"

    print(f"Sub-queries planned: {len(sub_queries)}")
    if sub_queries:
        for sq in sub_queries:
            print(f"  • {sq}")
    if replan_count:
        print(f"Replan attempts: {replan_count}")
    print(f"Chunks retrieved: {len(chunks)}")
    print(f"Confidence score: {confidence:.2f}")
    print()
    print("─" * 60)
    print(synthesis)
    print("─" * 60)

    if args.verbose and chunks:
        print("\n[Retrieved chunks]")
        seen: set[str] = set()
        for c in chunks:
            cid = str(c.get("chunk_id", ""))
            if cid in seen:
                continue
            seen.add(cid)
            sq = c.get("sub_query", "")
            title = c.get("paper_title", "Unknown")
            section = c.get("section", "")
            score = c.get("score", 0.0)
            text_preview = str(c.get("text", ""))[:200].replace("\n", " ")
            print(f"\n  [{sq}] {title} / {section} (score={score:.3f})")
            print(f"  {text_preview}...")


if __name__ == "__main__":
    main()
