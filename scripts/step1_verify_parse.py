"""
Step 1 verification — parse one paper and inspect the markdown output.

Checks for:
  - Section headers (# / ##)
  - LaTeX equations ($...$ or $$...$$)
  - Tables (| col | col |)

Usage:
    python scripts/step1_verify_parse.py
"""

import re
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.parser import PaperParser

console = Console()
PAPERS_DIR = Path("data/papers")


def inspect_markdown(markdown: str) -> None:
    headers = re.findall(r"^#{1,4} .+", markdown, re.MULTILINE)
    inline_math = re.findall(r"\$[^$\n]+\$", markdown)
    block_math = re.findall(r"\$\$[\s\S]+?\$\$", markdown)
    table_rows = re.findall(r"^\|.+\|", markdown, re.MULTILINE)

    stats = Table(show_header=True, header_style="bold cyan")
    stats.add_column("Feature", style="bold")
    stats.add_column("Count", justify="right")
    stats.add_column("Sample")

    stats.add_row(
        "Section headers",
        str(len(headers)),
        headers[0] if headers else "[dim]none[/dim]",
    )
    stats.add_row(
        "Inline math ($)",
        str(len(inline_math)),
        inline_math[0][:60] if inline_math else "[dim]none[/dim]",
    )
    stats.add_row(
        "Block math ($$)",
        str(len(block_math)),
        block_math[0][:60].replace("\n", " ") if block_math else "[dim]none[/dim]",
    )
    stats.add_row(
        "Table rows",
        str(len(table_rows)),
        table_rows[0][:60] if table_rows else "[dim]none[/dim]",
    )
    stats.add_row("Total chars", f"{len(markdown):,}", "")
    stats.add_row("Total lines", f"{markdown.count(chr(10)):,}", "")

    console.print(stats)

    console.print("\n[bold]First 60 lines of markdown:[/bold]")
    preview = "\n".join(markdown.splitlines()[:60])
    console.print(Panel(preview, border_style="dim"))

    console.print("\n[bold]All section headers found:[/bold]")
    for h in headers:
        console.print(f"  {h}")


def main() -> None:
    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdfs:
        console.print(f"[red]No PDFs found in {PAPERS_DIR}[/red]")
        sys.exit(1)

    target = pdfs[0]
    console.print(f"\n[bold]Inspecting:[/bold] {target.name}\n")

    parser = PaperParser()
    paper = parser.parse(target)

    inspect_markdown(paper.markdown)


if __name__ == "__main__":
    main()
