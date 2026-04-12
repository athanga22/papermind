"""
Step 1 — LlamaParse integration.

Parses PDFs to markdown via LlamaParse cloud API.
Caches results to data/parsed/ to avoid re-parsing on rerun.

Note: LlamaParse's load_data() manages its own event loop internally,
so the parser is synchronous — no asyncio needed at the call site.
"""

import hashlib
import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from llama_cloud_services.parse import LlamaParse, ResultType
from rich.console import Console

from ingestion.models import ParsedPaper

load_dotenv()

console = Console()

PARSED_CACHE_DIR = Path("data/parsed")


class PaperParser:
    """PDF → markdown via LlamaParse, with disk cache and retry."""

    def __init__(self) -> None:
        api_key = os.getenv("LLAMA_PARSE_API_KEY")
        if not api_key:
            raise ValueError("LLAMA_PARSE_API_KEY not set in environment")

        self._client = LlamaParse(
            api_key=api_key,
            result_type=ResultType.MD,
            verbose=False,
        )
        PARSED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def paper_id(pdf_path: Path) -> str:
        """Stable 12-char ID derived from filename."""
        return hashlib.md5(pdf_path.name.encode()).hexdigest()[:12]

    def _cache_path(self, paper_id: str) -> Path:
        return PARSED_CACHE_DIR / f"{paper_id}.md"

    def parse(self, pdf_path: Path, max_retries: int = 3) -> ParsedPaper:
        """
        Parse a single PDF to markdown.
        Returns cached result if already parsed.
        Raises RuntimeError after max_retries failures.
        """
        pid = self.paper_id(pdf_path)
        cache_path = self._cache_path(pid)

        if cache_path.exists():
            console.print(f"[dim]cache hit:[/dim] {pdf_path.name}")
            markdown = cache_path.read_text(encoding="utf-8")
            return ParsedPaper(
                paper_id=pid,
                file_name=pdf_path.name,
                file_path=str(pdf_path),
                markdown=markdown,
            )

        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                console.print(
                    f"Parsing [bold]{pdf_path.name}[/bold] "
                    f"(attempt {attempt}/{max_retries})..."
                )
                # load_data() returns List[Document]; each doc.text is markdown
                # for one page (or the whole file if split_by_page=False by default)
                docs = self._client.load_data(str(pdf_path))
                markdown = "\n\n".join(doc.text for doc in docs)

                if not markdown.strip():
                    raise ValueError("LlamaParse returned empty markdown")

                cache_path.write_text(markdown, encoding="utf-8")
                console.print(f"[green]✓[/green] {pdf_path.name}")

                return ParsedPaper(
                    paper_id=pid,
                    file_name=pdf_path.name,
                    file_path=str(pdf_path),
                    markdown=markdown,
                )

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait = 2 ** attempt
                    console.print(
                        f"[yellow]attempt {attempt} failed ({e}), "
                        f"retrying in {wait}s...[/yellow]"
                    )
                    time.sleep(wait)

        raise RuntimeError(
            f"Failed to parse {pdf_path.name} after {max_retries} attempts: {last_error}"
        )

    def parse_all(self, pdf_dir: Path, max_retries: int = 3) -> list[ParsedPaper]:
        """
        Parse all PDFs in a directory sequentially.
        Logs failures without stopping the batch.
        """
        pdfs = sorted(pdf_dir.glob("*.pdf"))
        if not pdfs:
            raise FileNotFoundError(f"No PDFs found in {pdf_dir}")

        results: list[ParsedPaper] = []
        failures: list[str] = []

        for pdf in pdfs:
            try:
                result = self.parse(pdf, max_retries=max_retries)
                results.append(result)
            except RuntimeError as e:
                console.print(f"[red]SKIP {pdf.name}: {e}[/red]")
                failures.append(pdf.name)

        if failures:
            console.print(
                f"\n[yellow]⚠ {len(failures)} file(s) failed: "
                f"{', '.join(failures)}[/yellow]"
            )

        console.print(f"\n[bold]Parsed {len(results)}/{len(pdfs)} papers.[/bold]")
        return results
