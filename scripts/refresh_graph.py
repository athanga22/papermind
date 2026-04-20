"""
Refresh Neo4j MENTIONS edges after re-ingestion.

Problem: Phase 1.5 re-chunked all papers from 256→512 tokens, generating new
sequential chunk_ids (e.g. abc123_0025 became abc123_0012). Neo4j still has
MENTIONS edges pointing at the old 256-tok chunk_ids, which no longer exist in
Qdrant. Graph retrieval silently drops them (payload cache miss), contributing
zero signal.

Fix:
  1. Delete all MENTIONS edges (they carry chunk_id — the only stale data)
  2. Reload current chunk payloads from Qdrant (source of truth)
  3. Re-run entity extraction (Haiku, sampled) with live chunk_ids
  4. Optionally re-run fix_neo4j synonym merging

CITES / CITES_PAPER / Reference nodes are NOT touched — they don't carry
chunk_ids and are still valid.

Cost estimate: 10 papers × ≤30 chunks × Haiku ≈ $0.03–$0.05

Usage:
    python scripts/refresh_graph.py              # full refresh
    python scripts/refresh_graph.py --dry-run    # show what would be cleared, no writes
    python scripts/refresh_graph.py --no-merge   # skip synonym merging at the end
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from ingestion.entity_extractor import EntityExtractor
from ingestion.models import Chunk
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

console = Console()

COLLECTION_NAME = "papers"
QDRANT_SCROLL_BATCH = 200


# ── Step 1: Load chunks from Qdrant ──────────────────────────────────────────

def load_chunks_from_qdrant() -> dict[str, list[Chunk]]:
    """
    Pull all chunk payloads from Qdrant and reconstruct Chunk objects.
    Groups by paper_id → used to feed EntityExtractor.process_paper().
    """
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
    )

    chunks_by_paper: dict[str, list[Chunk]] = defaultdict(list)
    offset = None
    total = 0

    console.print("Loading chunks from Qdrant...", end=" ")
    while True:
        results, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=QDRANT_SCROLL_BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for pt in results:
            p = pt.payload
            chunk = Chunk(
                chunk_id    = p["chunk_id"],
                paper_id    = p["paper_id"],
                paper_title = p.get("paper_title", ""),
                authors     = p.get("authors", []),
                year        = p.get("year"),
                section     = p.get("section", ""),
                chunk_index = p.get("chunk_index", 0),
                text        = p.get("text", ""),
                is_table    = p.get("is_table", False),
            )
            chunks_by_paper[chunk.paper_id].append(chunk)
            total += 1

        offset = next_offset
        if offset is None:
            break

    console.print(f"[green]{total} chunks across {len(chunks_by_paper)} papers[/green]")
    return dict(chunks_by_paper)


# ── Step 2: Clear stale MENTIONS ─────────────────────────────────────────────

def clear_mentions(driver, dry_run: bool = False) -> int:
    """Delete all MENTIONS edges from Neo4j. Returns count deleted."""
    with driver.session() as session:
        count = session.run(
            "MATCH ()-[r:MENTIONS]->() RETURN count(r) AS n"
        ).single()["n"]

        if dry_run:
            console.print(f"[yellow]DRY RUN:[/yellow] Would delete {count} MENTIONS edges")
            return count

        # Delete in batches to avoid OOM on large graphs
        deleted = 0
        while True:
            result = session.run(
                "MATCH ()-[r:MENTIONS]->() WITH r LIMIT 1000 DELETE r RETURN count(r) AS n"
            ).single()["n"]
            deleted += result
            if result < 1000:
                break

        console.print(f"Deleted [bold]{deleted}[/bold] stale MENTIONS edges")
        return deleted


# ── Step 3: Re-run entity extraction ─────────────────────────────────────────

def refresh_entity_mentions(
    chunks_by_paper: dict[str, list[Chunk]],
    dry_run: bool = False,
) -> None:
    """Re-extract entities for all papers using live Qdrant chunks."""
    if dry_run:
        console.print(
            f"[yellow]DRY RUN:[/yellow] Would extract entities for "
            f"{len(chunks_by_paper)} papers"
        )
        return

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Paper ID")
    table.add_column("Chunks")
    table.add_column("Mentions", justify="right")

    total_mentions = 0

    with EntityExtractor() as extractor:
        for paper_id, chunks in sorted(chunks_by_paper.items()):
            # Sort by chunk_index so sampling is evenly spread
            chunks.sort(key=lambda c: c.chunk_index)
            mentions = extractor.process_paper(chunks, full_paper=False, verbose=False)
            table.add_row(paper_id, str(len(chunks)), str(mentions))
            total_mentions += mentions
            console.print(f"  [dim]{paper_id}[/dim] → {mentions} mentions")

    console.print(table)
    console.print(f"\nTotal new MENTIONS written: [bold green]{total_mentions}[/bold green]")


# ── Step 4: Optional synonym merge ───────────────────────────────────────────

def run_synonym_merge(dry_run: bool = False) -> None:
    """Re-apply synonym and type-conflict merging from scripts/fix_neo4j.py."""
    if dry_run:
        console.print("[yellow]DRY RUN:[/yellow] Would run synonym/type merging")
        return

    console.print("\nRunning synonym + type-conflict merging...")
    # Import and reuse existing fix_neo4j logic
    sys.path.insert(0, str(Path(__file__).parent))
    from fix_neo4j import merge_entities_by_synonym, merge_same_name_diff_type
    from neo4j import GraphDatabase as GDB

    driver = GDB.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    with driver.session() as session:
        n1 = merge_entities_by_synonym(session)
        n2 = merge_same_name_diff_type(session)
    driver.close()
    console.print(f"  Synonym merges: {n1} | Type-conflict merges: {n2}")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_graph_summary(driver) -> None:
    with driver.session() as session:
        console.print("\n[bold]Graph state after refresh:[/bold]")
        for label in ["Paper", "Entity", "Reference"]:
            n = session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
            console.print(f"  {label:12s}: {n}")
        for rel in ["MENTIONS", "CITES", "CITES_PAPER"]:
            n = session.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c").single()["c"]
            console.print(f"  {rel:12s}: {n}")
        shared = session.run("""
            MATCH (p:Paper)-[:MENTIONS]->(e:Entity)
            WITH e, count(DISTINCT p) AS cnt
            WHERE cnt >= 2
            RETURN count(e) AS n
        """).single()["n"]
        console.print(f"\n  Entities shared by 2+ papers: [bold green]{shared}[/bold green]")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh Neo4j MENTIONS edges")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without writing anything")
    parser.add_argument("--no-merge", action="store_true",
                        help="Skip synonym/type-conflict merging at the end")
    args = parser.parse_args()

    console.print("\n[bold cyan]Neo4j MENTIONS Refresh[/bold cyan]")
    if args.dry_run:
        console.print("[yellow]DRY RUN MODE — no writes[/yellow]\n")

    t0 = time.time()

    # 1. Load current chunks from Qdrant
    chunks_by_paper = load_chunks_from_qdrant()

    # 2. Connect to Neo4j
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )

    # 3. Clear stale MENTIONS
    console.print("\n[bold]Step 1:[/bold] Clearing stale MENTIONS edges...")
    clear_mentions(driver, dry_run=args.dry_run)

    # 4. Re-run entity extraction
    console.print("\n[bold]Step 2:[/bold] Re-extracting entities (Haiku, sampled)...")
    refresh_entity_mentions(chunks_by_paper, dry_run=args.dry_run)

    # 5. Synonym merge
    if not args.no_merge:
        run_synonym_merge(dry_run=args.dry_run)

    # 6. Summary
    if not args.dry_run:
        print_graph_summary(driver)

    driver.close()
    elapsed = time.time() - t0
    console.print(f"\n[bold green]Done[/bold green] in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
