"""
CLI: run a dense similarity search and print the top results.

Usage:
    python -m papermind.scripts.query "how does reinforcement learning improve reasoning?"
    python -m papermind.scripts.query "test-time compute scaling" --top-k 3
"""

from __future__ import annotations

import click
from rich.console import Console
from rich.panel import Panel

from ..retrieval.store import get_store

console = Console()


@click.command()
@click.argument("query")
@click.option("--top-k", default=5, show_default=True)
def main(query: str, top_k: int) -> None:
    store = get_store()
    results = store.search(query, top_k=top_k)

    if not results:
        console.print("[yellow]No results. Have you run make ingest yet?[/yellow]")
        return

    console.print(f"\n[bold]Query:[/bold] {query}\n")

    for rc in results:
        c = rc.chunk
        console.print(Panel(
            f"[dim]{c.text[:400]}{'...' if len(c.text) > 400 else ''}[/dim]",
            title=f"[cyan][{rc.rank}] {c.paper_title}[/cyan]  "
                  f"[yellow]{c.section.title()}[/yellow]  p.{c.page_number}  "
                  f"score=[green]{rc.score:.4f}[/green]",
            expand=False,
        ))


if __name__ == "__main__":
    main()
