"""
CLI: parse PDFs and index them into Qdrant.

Usage:
    python -m papermind.scripts.ingest --pdf-dir ./data/papers
    python -m papermind.scripts.ingest --pdf-dir ./data/papers --recreate
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ..ingestion.chunker import chunk_paper
from ..ingestion.parser import parse_pdf
from ..retrieval.store import get_store

console = Console()
logging.basicConfig(level=logging.WARNING)


@click.command()
@click.option("--pdf-dir", required=True, type=click.Path(exists=True))
@click.option("--recreate", is_flag=True, default=False, help="Drop and recreate the collection")
@click.option("--verbose", "-v", is_flag=True, default=False)
def main(pdf_dir: str, recreate: bool, verbose: bool) -> None:
    if verbose:
        logging.basicConfig(level=logging.INFO, force=True)

    pdfs = list(Path(pdf_dir).glob("**/*.pdf"))
    if not pdfs:
        console.print(f"[red]No PDFs found in {pdf_dir}[/red]")
        sys.exit(1)

    console.print(f"\n[bold]PaperMind Ingest[/bold] — {len(pdfs)} PDF(s)\n")

    store = get_store()
    store.ensure_collection(recreate=recreate)

    stats: list[dict] = []  # type: ignore[type-arg]

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        for pdf in pdfs:
            task = progress.add_task(f"[cyan]{pdf.name}[/cyan]", total=None)
            try:
                paper = parse_pdf(pdf)
                chunks = chunk_paper(paper)
                store.upsert_chunks(chunks)
                stats.append({
                    "file": pdf.name,
                    "title": paper.title[:50],
                    "sections": len(paper.sections),
                    "chunks": len(chunks),
                    "status": "[green]OK[/green]",
                })
            except Exception as e:
                stats.append({
                    "file": pdf.name,
                    "title": "-",
                    "sections": 0,
                    "chunks": 0,
                    "status": f"[red]FAILED: {e}[/red]",
                })
            progress.remove_task(task)

    table = Table(title="Ingestion Summary", show_lines=True)
    table.add_column("File", style="cyan")
    table.add_column("Title")
    table.add_column("Sections", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Status")

    total_chunks = 0
    for s in stats:
        table.add_row(s["file"], s["title"], str(s["sections"]), str(s["chunks"]), s["status"])
        total_chunks += s["chunks"]

    console.print(table)
    console.print(f"\n[bold]Total chunks indexed: {total_chunks}[/bold]")
    console.print(f"Qdrant collection total: {store.count()} chunks\n")


if __name__ == "__main__":
    main()
