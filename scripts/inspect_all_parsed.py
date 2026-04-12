"""
Quick inspection of all 10 cached markdown files.
Reports headers, math, tables, and size per paper.

Usage:
    python scripts/inspect_all_parsed.py
"""

import re
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.parser import PaperParser

console = Console()
PAPERS_DIR = Path("data/papers")
PARSED_DIR = Path("data/parsed")


def inspect(markdown: str) -> dict:
    headers = re.findall(r"^#{1,4} .+", markdown, re.MULTILINE)
    inline_math = re.findall(r"\$[^$\n]+\$", markdown)
    block_math = re.findall(r"\$\$[\s\S]+?\$\$", markdown)
    table_rows = re.findall(r"^\|.+\|", markdown, re.MULTILINE)
    return {
        "headers": len(headers),
        "inline_math": len(inline_math),
        "block_math": len(block_math),
        "tables": len(table_rows),
        "chars": len(markdown),
        "lines": markdown.count("\n"),
    }


def main() -> None:
    parser = PaperParser()
    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))

    table = Table(show_header=True, header_style="bold cyan", show_lines=True)
    table.add_column("Paper", style="bold", max_width=25)
    table.add_column("Headers", justify="right")
    table.add_column("Inline $", justify="right")
    table.add_column("Block $$", justify="right")
    table.add_column("Table rows", justify="right")
    table.add_column("Chars", justify="right")
    table.add_column("Notes")

    for pdf in pdfs:
        paper = parser.parse(pdf)
        s = inspect(paper.markdown)

        notes = []
        if s["headers"] < 5:
            notes.append("⚠ few headers")
        if s["chars"] < 10_000:
            notes.append("⚠ short")
        if s["block_math"] > 10:
            notes.append("math-heavy")
        if s["tables"] > 50:
            notes.append("table-heavy")

        table.add_row(
            pdf.name,
            str(s["headers"]),
            str(s["inline_math"]),
            str(s["block_math"]),
            str(s["tables"]),
            f"{s['chars']:,}",
            ", ".join(notes) if notes else "ok",
        )

    console.print(table)


if __name__ == "__main__":
    main()
