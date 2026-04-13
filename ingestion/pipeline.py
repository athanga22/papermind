"""
Phase 1 full ingestion pipeline.

Runs Steps 1-7 for all PDFs in data/papers/:
  1. Parse PDF → markdown (LlamaParse, cached)
  2. Chunk markdown (section-aware, no LLM)
  3. Metadata tagging (regex, no LLM)
  4. Dense embedding → Qdrant
  5. BM25 index → disk
  6. Entity extraction → Neo4j  (Haiku, sampled)
  7. Citation graph → Neo4j

Usage:
    python -m ingestion.pipeline            # full run
    python -m ingestion.pipeline --no-neo4j # skip Steps 6-7 (no Neo4j needed)
    python -m ingestion.pipeline --dry-run  # parse + chunk only, no external calls
"""

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Resolve .env relative to the repo root (one level above this package)
_REPO_ROOT = Path(__file__).parent.parent
load_dotenv(_REPO_ROOT / ".env", override=True)

from ingestion.bm25_index import BM25Index
from ingestion.chunker import PaperChunker
from ingestion.citation_graph import CitationGraphWriter
from ingestion.embedder import ChunkEmbedder
from ingestion.entity_extractor import EntityExtractor
from ingestion.metadata import extract_bibliography, extract_paper_metadata
from ingestion.models import Chunk
from ingestion.parser import PaperParser

PAPERS_DIR = Path("data/papers")
console = Console()


def run_pipeline(
    skip_neo4j: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
) -> None:
    t0 = time.time()

    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdfs:
        console.print(f"[red]No PDFs found in {PAPERS_DIR}[/red]")
        sys.exit(1)

    console.print(f"\n[bold cyan]PaperMind Ingestion Pipeline[/bold cyan]")
    console.print(f"Papers: {len(pdfs)} | dry_run={dry_run} | skip_neo4j={skip_neo4j}\n")

    # ── Step 1: Parse ─────────────────────────────────────────────────────────
    console.print("[bold]Step 1:[/bold] Parsing PDFs (LlamaParse, cached)...")
    parser = PaperParser()
    papers = []
    for pdf in pdfs:
        paper = parser.parse(pdf)
        papers.append(paper)
        console.print(f"  ✓ {pdf.name} ({len(paper.markdown):,} chars)")

    # ── Steps 2-3: Chunk + metadata ───────────────────────────────────────────
    console.print(f"\n[bold]Steps 2-3:[/bold] Chunking + metadata extraction...")
    chunker = PaperChunker(chunk_size=512, chunk_overlap=64)
    all_chunks: list[Chunk] = []
    paper_chunks: dict[str, list[Chunk]] = {}  # paper_id → chunks
    bibs = []

    stats_table = Table(show_header=True, header_style="bold cyan")
    stats_table.add_column("Paper", max_width=22)
    stats_table.add_column("Title", max_width=35)
    stats_table.add_column("Authors")
    stats_table.add_column("Year", justify="right")
    stats_table.add_column("Chunks", justify="right")
    stats_table.add_column("Refs", justify="right")

    for paper in papers:
        meta = extract_paper_metadata(paper)
        bib = extract_bibliography(paper)
        bibs.append(bib)

        chunks = chunker.chunk(
            paper,
            title=meta["title"],
            authors=meta["authors"],
            year=meta["year"],
        )
        paper_chunks[paper.paper_id] = chunks
        all_chunks.extend(chunks)

        stats_table.add_row(
            paper.file_name,
            meta["title"][:35],
            ", ".join(meta["authors"][:2]) + (f" +{len(meta['authors'])-2}" if len(meta["authors"]) > 2 else ""),
            str(meta["year"]) if meta["year"] else "?",
            str(len(chunks)),
            str(len(bib.references)),
        )

    console.print(stats_table)
    console.print(f"\nTotal chunks: [bold]{len(all_chunks)}[/bold]")

    if dry_run:
        console.print("\n[yellow]Dry run — stopping before external writes.[/yellow]")
        return

    # ── Step 4: Dense embedding → Qdrant ─────────────────────────────────────
    console.print(f"\n[bold]Step 4:[/bold] Embedding {len(all_chunks)} chunks → Qdrant...")
    embedder = ChunkEmbedder()
    total_upserted = embedder.embed_and_store(all_chunks)
    console.print(f"  ✓ Upserted {total_upserted} points | collection total: {embedder.collection_count()}")

    # ── Step 5: BM25 index ────────────────────────────────────────────────────
    console.print(f"\n[bold]Step 5:[/bold] Building BM25 index...")
    bm25 = BM25Index()
    bm25.build(all_chunks, show_progress=False)
    console.print(f"  ✓ BM25 index built — {bm25.size} chunks indexed")

    if skip_neo4j:
        console.print("\n[yellow]Skipping Steps 6-7 (--no-neo4j)[/yellow]")
    else:
        # ── Step 6: Entity extraction → Neo4j ─────────────────────────────────
        console.print(f"\n[bold]Step 6:[/bold] Entity extraction (Haiku, sampled)...")
        with EntityExtractor() as extractor:
            total_mentions = 0
            for paper in papers:
                chunks = paper_chunks[paper.paper_id]
                mentions = extractor.process_paper(chunks, full_paper=False, verbose=verbose)
                console.print(f"  ✓ {paper.file_name}: {mentions} entity mentions")
                total_mentions += mentions
        console.print(f"  Total mentions: {total_mentions}")

        # ── Step 7: Citation graph ─────────────────────────────────────────────
        console.print(f"\n[bold]Step 7:[/bold] Building citation graph...")
        with CitationGraphWriter() as cg_writer:
            total_refs = 0
            for bib in bibs:
                written = cg_writer.write_bibliography(bib.paper_id, bib)
                total_refs += written
        console.print(f"  ✓ {total_refs} reference nodes written")

    elapsed = time.time() - t0
    console.print(Panel(
        f"[bold green]Phase 1 complete[/bold green] in {elapsed:.1f}s\n"
        f"Papers: {len(papers)} | Chunks: {len(all_chunks)} | "
        f"Qdrant: {total_upserted if not dry_run else 'skipped'} points",
        border_style="green",
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description="PaperMind ingestion pipeline")
    parser.add_argument("--no-neo4j", action="store_true", help="Skip Steps 6-7")
    parser.add_argument("--dry-run", action="store_true", help="Parse + chunk only")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    run_pipeline(
        skip_neo4j=args.no_neo4j,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
