"""
Step 2 verification — inspect chunk output for one paper.

Checks:
  - Chunk count and type distribution (text vs table)
  - Average chunk size in characters
  - Section distribution
  - No chunks shorter than 50 chars
  - Sample chunks

Usage:
    python scripts/step2_verify_chunking.py
"""

import sys
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.chunker import PaperChunker
from ingestion.parser import PaperParser

console = Console()
PAPERS_DIR = Path("data/papers")

# Placeholder metadata — Step 3 will extract these properly from the markdown
_PLACEHOLDER = {
    "title": "Unknown Title",
    "authors": ["Unknown"],
    "year": None,
}


def main() -> None:
    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdfs:
        console.print(f"[red]No PDFs found in {PAPERS_DIR}[/red]")
        sys.exit(1)

    target = pdfs[0]
    console.print(f"\n[bold]Chunking:[/bold] {target.name}\n")

    parser = PaperParser()
    paper = parser.parse(target)

    chunker = PaperChunker(chunk_size=256, chunk_overlap=20)
    chunks = chunker.chunk(
        paper,
        title=_PLACEHOLDER["title"],
        authors=_PLACEHOLDER["authors"],
        year=_PLACEHOLDER["year"],
    )

    # ── Stats ─────────────────────────────────────────────────────────────────
    text_chunks = [c for c in chunks if not c.is_table]
    table_chunks = [c for c in chunks if c.is_table]
    math_chunks = [c for c in chunks if c.contains_math]

    avg_text_len = (
        sum(len(c.text) for c in text_chunks) / len(text_chunks)
        if text_chunks else 0
    )
    avg_table_len = (
        sum(len(c.text) for c in table_chunks) / len(table_chunks)
        if table_chunks else 0
    )
    tiny = [c for c in chunks if len(c.text) < 50]

    stats = Table(show_header=True, header_style="bold cyan")
    stats.add_column("Metric", style="bold")
    stats.add_column("Value", justify="right")

    stats.add_row("Total chunks", str(len(chunks)))
    stats.add_row("Text chunks", str(len(text_chunks)))
    stats.add_row("Table chunks", str(len(table_chunks)))
    stats.add_row("Math chunks", str(len(math_chunks)))
    stats.add_row("Avg text chunk (chars)", f"{avg_text_len:.0f}")
    stats.add_row("Avg table chunk (chars)", f"{avg_table_len:.0f}")
    stats.add_row("Tiny chunks (<50 chars)", str(len(tiny)))

    console.print(stats)

    # ── Section distribution ──────────────────────────────────────────────────
    section_counts = Counter(c.section for c in chunks)
    console.print(f"\n[bold]Section distribution ({len(section_counts)} sections):[/bold]")
    sec_table = Table(show_header=True, header_style="bold cyan")
    sec_table.add_column("Section", max_width=50)
    sec_table.add_column("Chunks", justify="right")
    for section, count in section_counts.most_common(15):
        sec_table.add_row(section, str(count))
    console.print(sec_table)

    # ── Sample chunks ─────────────────────────────────────────────────────────
    console.print("\n[bold]Sample text chunk:[/bold]")
    if text_chunks:
        c = text_chunks[2] if len(text_chunks) > 2 else text_chunks[0]
        console.print(Panel(
            f"[dim]section:[/dim] {c.section}\n"
            f"[dim]chars:[/dim] {len(c.text)}\n\n"
            f"{c.text[:400]}",
            border_style="dim",
        ))

    console.print("\n[bold]Sample table chunk:[/bold]")
    if table_chunks:
        c = table_chunks[0]
        console.print(Panel(
            f"[dim]section:[/dim] {c.section}\n"
            f"[dim]chars:[/dim] {len(c.text)}\n\n"
            f"{c.text[:400]}",
            border_style="dim",
        ))

    if tiny:
        console.print(f"\n[yellow]⚠ {len(tiny)} tiny chunks found — investigate:[/yellow]")
        for c in tiny[:3]:
            console.print(f"  [{c.chunk_id}] '{c.text[:80]}'")
    else:
        console.print("\n[green]✓ No tiny chunks — filter working correctly.[/green]")


if __name__ == "__main__":
    main()
