"""
Parse all 10 papers via LlamaParse and cache to data/parsed/.
Run once — subsequent runs read from cache at zero API cost.

Usage:
    python scripts/parse_all_papers.py
"""

import sys
from pathlib import Path

from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.parser import PaperParser

console = Console()
PAPERS_DIR = Path("data/papers")


def main() -> None:
    parser = PaperParser()
    papers = parser.parse_all(PAPERS_DIR)
    console.print(f"\n[green bold]Done. {len(papers)} papers cached to data/parsed/[/green bold]")


if __name__ == "__main__":
    main()
