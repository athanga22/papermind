"""
Step 3 verification — metadata and bibliography extraction across all papers.

Checks per paper:
  - Title extracted (not "Unknown Title")
  - Authors extracted (not ["Unknown"])
  - Year extracted (valid 20xx)
  - Bibliography count

Usage:
    python scripts/step3_verify_metadata.py
"""

import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.metadata import extract_bibliography, extract_paper_metadata
from ingestion.parser import PaperParser

console = Console()
PAPERS_DIR = Path("data/papers")


def main() -> None:
    parser = PaperParser()
    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdfs:
        console.print(f"[red]No PDFs found in {PAPERS_DIR}[/red]")
        sys.exit(1)

    table = Table(show_header=True, header_style="bold cyan", show_lines=True)
    table.add_column("Paper", max_width=22, style="bold")
    table.add_column("Year", justify="right")
    table.add_column("Title (first 40)", max_width=42)
    table.add_column("Authors", max_width=35)
    table.add_column("Refs", justify="right")
    table.add_column("Issues")

    for pdf in pdfs:
        paper = parser.parse(pdf)
        meta = extract_paper_metadata(paper)
        bib = extract_bibliography(paper)

        issues = []
        if meta["title"] == "Unknown Title":
            issues.append("no title")
        if meta["authors"] == ["Unknown"]:
            issues.append("no authors")
        if meta["year"] is None:
            issues.append("no year")
        if len(bib.references) == 0:
            issues.append("no refs")

        authors_str = ", ".join(meta["authors"][:3])
        if len(meta["authors"]) > 3:
            authors_str += f" +{len(meta['authors']) - 3}"

        table.add_row(
            pdf.name,
            str(meta["year"]) if meta["year"] else "[red]?[/red]",
            meta["title"][:40],
            authors_str,
            str(len(bib.references)),
            "[red]" + ", ".join(issues) + "[/red]" if issues else "[green]ok[/green]",
        )

    console.print(table)

    # ── Deep dive on first paper ──────────────────────────────────────────────
    paper = parser.parse(pdfs[0])
    meta = extract_paper_metadata(paper)
    bib = extract_bibliography(paper)

    console.print(f"\n[bold]Deep dive:[/bold] {pdfs[0].name}")
    console.print(Panel(
        f"[bold]Title:[/bold] {meta['title']}\n"
        f"[bold]Year:[/bold]  {meta['year']}\n"
        f"[bold]Authors ({len(meta['authors'])}):[/bold]\n"
        + "\n".join(f"  • {a}" for a in meta["authors"][:10]),
        title="Metadata",
        border_style="dim",
    ))

    if bib.references:
        console.print(f"\n[bold]First 3 references:[/bold]")
        for ref in bib.references[:3]:
            console.print(f"  [dim]{ref[:120]}[/dim]")
    else:
        console.print("\n[yellow]⚠ No references extracted[/yellow]")


if __name__ == "__main__":
    main()
