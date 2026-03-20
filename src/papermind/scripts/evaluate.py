"""
CLI: run RAGAS evaluation against the test set.

Usage:
    python -m papermind.scripts.evaluate --test-set data/test_set/questions.json --stage baseline
    python -m papermind.scripts.evaluate --test-set data/test_set/questions.json --stage hybrid
    python -m papermind.scripts.evaluate --history   # print progression table
"""

from __future__ import annotations

import logging
import sys

import click
from rich.console import Console
from rich.table import Table

from ..eval.ragas_eval import EvalResult, load_results, run_evaluation
from ..eval.test_set import load_test_set

console = Console()
logging.basicConfig(level=logging.WARNING)


@click.group()
def main() -> None:
    pass


@main.command()
@click.option("--test-set", required=True, type=click.Path(exists=True))
@click.option("--stage", default="unnamed", help="Label for this eval run (e.g. baseline, hybrid)")
@click.option("--output-dir", default="eval_results", show_default=True)
@click.option("--verbose", "-v", is_flag=True, default=False)
def run(test_set: str, stage: str, output_dir: str, verbose: bool) -> None:
    """Run RAGAS evaluation for a single build stage."""
    if verbose:
        logging.basicConfig(level=logging.INFO, force=True)

    ts = load_test_set(test_set)
    console.print(f"\n[bold]Evaluating stage: [cyan]{stage}[/cyan][/bold] — {len(ts)} questions\n")

    result = run_evaluation(ts, stage_name=stage, output_dir=output_dir)

    console.print(f"[green]Done.[/green] Results saved to {output_dir}/")


@main.command()
@click.option("--output-dir", default="eval_results", show_default=True)
def history(output_dir: str) -> None:
    """Print the RAGAS progression table across all saved eval runs."""
    results = load_results(output_dir)
    if not results:
        console.print("[yellow]No eval results found.[/yellow]")
        sys.exit(0)

    table = Table(title="RAGAS Progression", show_lines=True)
    table.add_column("Stage", style="cyan")
    table.add_column("Ctx Precision", justify="right")
    table.add_column("Ctx Recall", justify="right")
    table.add_column("Faithfulness", justify="right")
    table.add_column("Answer Rel.", justify="right")
    table.add_column("N", justify="right")

    for r in results:
        table.add_row(
            r.stage,
            f"{r.context_precision:.4f}",
            f"{r.context_recall:.4f}",
            f"{r.faithfulness:.4f}",
            f"{r.answer_relevancy:.4f}",
            str(r.n_questions),
        )

    console.print(table)


if __name__ == "__main__":
    main()
