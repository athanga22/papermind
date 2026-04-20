"""
Suggest chunk_ids for golden-set rows (especially cross-paper) by dense search in Qdrant.

For each paper_id, embeds the question (optionally augmented with a short hint) and runs
vector search restricted to that paper. Pick chunks that, together, could ground the
reference answer; verify by reading the printed excerpts.

Cross-paper rows in this repo often use chunk_id: null; you can add a "chunk_ids" list
(two ids — one primary chunk per paper) while keeping chunk_id null, or adopt another
convention your eval tooling understands.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env", override=True)

EMBEDDING_MODEL = "text-embedding-3-small"


def _embed(client: OpenAI, text: str) -> list[float]:
    return client.embeddings.create(input=[text], model=EMBEDDING_MODEL).data[0].embedding


def search_paper(
    qdrant: QdrantClient,
    oai: OpenAI,
    *,
    paper_id: str,
    query_text: str,
    limit: int,
) -> list:
    return qdrant.query_points(
        collection_name="papers",
        query=_embed(oai, query_text),
        query_filter=Filter(
            must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
        ),
        limit=limit,
        with_payload=True,
    ).points


def _print_hits(paper_id: str, hits: list, preview_chars: int) -> list[str]:
    ids: list[str] = []
    print(f"\n=== paper_id={paper_id}  ({len(hits)} hits) ===")
    for rank, h in enumerate(hits, 1):
        p = h.payload
        cid = p.get("chunk_id", "?")
        ids.append(cid)
        sec = p.get("section", "?")
        text = (p.get("text") or "")[:preview_chars].replace("\n", " ")
        print(f"  #{rank}  score={h.score:.4f}  chunk_id={cid}  [{sec}]")
        print(f"       {text}{'…' if len(p.get('text') or '') > preview_chars else ''}")
    return ids


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-q",
        "--question",
        help="Evaluation question text (recommended: same string as in the golden JSON).",
    )
    p.add_argument(
        "--paper-id",
        action="append",
        dest="paper_ids",
        help="Paper id (repeat for each paper in a cross-paper item).",
    )
    p.add_argument(
        "--hint",
        help="Optional short phrase appended to the question before embedding "
        "(narrower retrieval, like regenerate_cross_paper search_query).",
    )
    p.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k chunks per paper (default 5).",
    )
    p.add_argument(
        "--preview",
        type=int,
        default=280,
        help="Max characters of chunk text to print per hit.",
    )
    p.add_argument(
        "--from-json",
        type=Path,
        help="Load one record from a JSON array (e.g. test set/cross_paper.json).",
    )
    p.add_argument(
        "--index",
        type=int,
        default=0,
        help="Record index when using --from-json (default 0).",
    )
    args = p.parse_args()

    question = args.question
    paper_ids = list(args.paper_ids or [])

    if args.from_json:
        path = args.from_json
        if not path.is_file():
            print(f"Not a file: {path}", file=sys.stderr)
            sys.exit(1)
        with path.open() as f:
            data = json.load(f)
        if not isinstance(data, list):
            print("--from-json expects a JSON array", file=sys.stderr)
            sys.exit(1)
        rec = data[args.index]
        question = question or rec.get("question")
        paper_ids = paper_ids or list(rec.get("paper_ids") or [])

    if not question or not paper_ids:
        p.print_help()
        print(
            "\nExample:\n"
            "  python scripts/c2t.py --from-json 'test set/cross_paper.json' --index 0\n"
            "  python scripts/c2t.py -q 'Your question…' --paper-id abc --paper-id def",
            file=sys.stderr,
        )
        sys.exit(1)

    query_text = question.strip()
    if args.hint:
        query_text = f"{query_text}\n{args.hint.strip()}"

    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant = QdrantClient(host=host, port=port)
    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    print(f"Query ({len(query_text)} chars):\n{query_text[:500]}{'…' if len(query_text) > 500 else ''}")

    top1_ids: list[str] = []
    for pid in paper_ids:
        hits = search_paper(qdrant, oai, paper_id=pid, query_text=query_text, limit=args.k)
        ids = _print_hits(pid, hits, args.preview)
        if ids:
            top1_ids.append(ids[0])

    if len(top1_ids) == len(paper_ids):
        print("\n--- Suggested (top-1 per paper, verify before committing) ---")
        print(json.dumps({"chunk_ids": top1_ids}, indent=2))


if __name__ == "__main__":
    main()
