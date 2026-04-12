"""
Round 1 retrieval evaluation against the silver QA set.

For each question:
  - Runs trimodal retrieval (dense + BM25 + graph)
  - Also runs each mode in isolation for comparison
  - For single-paper questions: checks if known chunk_id appears in top-k
  - For cross-paper questions: shows top results for manual inspection

Outputs:
  - Console summary table
  - docs/eval_round1.json  (full results for future reference)

Usage:
    python scripts/run_retrieval_eval.py
"""

import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv(Path(__file__).parent.parent / ".env", override=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

from query.retriever import TrimodalRetriever

SILVER_PATH = Path("docs/golden_set_silver.json")
OUT_PATH    = Path("docs/eval_round1.json")
K_VALUES    = [1, 3, 5, 10]

console = Console()


def hit_at_k(retrieved_ids: list[str], target_id: str, k: int) -> bool:
    return target_id in retrieved_ids[:k]


def reciprocal_rank(retrieved_ids: list[str], target_id: str) -> float:
    try:
        return 1.0 / (retrieved_ids.index(target_id) + 1)
    except ValueError:
        return 0.0


def run_eval():
    with open(SILVER_PATH) as f:
        silver = json.load(f)

    console.print(f"\n[bold cyan]Retrieval Eval — Round 1[/bold cyan]")
    console.print(f"Questions: {len(silver)} | K values: {K_VALUES}\n")

    results = []

    with TrimodalRetriever() as retriever:
        for i, qa in enumerate(silver, 1):
            q         = qa["question"]
            target_id = qa.get("chunk_id")       # None for cross-paper
            q_type    = qa.get("type", "?")
            source    = qa.get("source", "?")
            paper     = qa.get("paper_title", "cross-paper")

            console.print(f"[dim][{i:02d}/{len(silver)}][/dim] {q[:80]}...")

            # ── Run all 4 modes ───────────────────────────────────────────────
            modes = {
                "trimodal": dict(use_dense=True,  use_bm25=True,  use_graph=True),
                "dense":    dict(use_dense=True,  use_bm25=False, use_graph=False),
                "bm25":     dict(use_dense=False, use_bm25=True,  use_graph=False),
                "graph":    dict(use_dense=False, use_bm25=False, use_graph=True),
            }

            mode_results = {}
            for mode_name, kwargs in modes.items():
                hits = retriever.retrieve(q, top_k=10, **kwargs)
                ids  = [h.chunk_id for h in hits]
                mode_results[mode_name] = {
                    "chunk_ids": ids,
                    "top3": [
                        {
                            "chunk_id": h.chunk_id,
                            "paper":    h.paper_title[:50],
                            "section":  h.section,
                            "text":     h.text[:150],
                            "score":    round(h.score, 4),
                            "sources":  h.sources,
                        }
                        for h in hits[:3]
                    ],
                }
                if target_id:
                    mode_results[mode_name]["hits"] = {
                        f"hit@{k}": hit_at_k(ids, target_id, k) for k in K_VALUES
                    }
                    mode_results[mode_name]["rr"] = reciprocal_rank(ids, target_id)

            results.append({
                "id":          i,
                "question":    q,
                "answer":      qa.get("answer", ""),
                "type":        q_type,
                "source":      source,
                "paper_title": paper,
                "chunk_id":    target_id,
                "paper_ids":   qa.get("paper_ids", []),
                "modes":       mode_results,
            })

            time.sleep(0.2)  # avoid rate limiting

    # ── Compute aggregate metrics ─────────────────────────────────────────────
    single = [r for r in results if r["source"] == "single_paper"]

    console.print(f"\n{'='*70}")
    console.print(f"[bold]RESULTS SUMMARY[/bold]")
    console.print(f"{'='*70}\n")

    # Per-mode Hit Rate and MRR table
    metrics_table = Table(show_header=True, header_style="bold cyan")
    metrics_table.add_column("Mode",      width=12)
    metrics_table.add_column("Hit@1",  justify="right")
    metrics_table.add_column("Hit@3",  justify="right")
    metrics_table.add_column("Hit@5",  justify="right")
    metrics_table.add_column("Hit@10", justify="right")
    metrics_table.add_column("MRR",    justify="right")

    for mode in ["trimodal", "dense", "bm25", "graph"]:
        h = {f"hit@{k}": 0 for k in K_VALUES}
        rr_sum = 0.0
        for r in single:
            m = r["modes"][mode]
            for k in K_VALUES:
                if m.get("hits", {}).get(f"hit@{k}"):
                    h[f"hit@{k}"] += 1
            rr_sum += m.get("rr", 0.0)

        n = len(single)
        style = "bold green" if mode == "trimodal" else ""
        metrics_table.add_row(
            f"[{style}]{mode}[/{style}]" if style else mode,
            *[f"[{style}]{h[f'hit@{k}']/n:.0%}[/{style}]" if style
              else f"{h[f'hit@{k}']/n:.0%}" for k in K_VALUES],
            f"[{style}]{rr_sum/n:.3f}[/{style}]" if style else f"{rr_sum/n:.3f}",
        )

    console.print(metrics_table)
    console.print(f"  (n={len(single)} single-paper questions with known chunk_ids)\n")

    # ── Per-question breakdown ────────────────────────────────────────────────
    detail_table = Table(show_header=True, header_style="bold", show_lines=True)
    detail_table.add_column("#",       width=3,  justify="right")
    detail_table.add_column("Type",    width=12)
    detail_table.add_column("Question",           max_width=38)
    detail_table.add_column("tri\nH@5", width=5, justify="center")
    detail_table.add_column("den\nH@5", width=5, justify="center")
    detail_table.add_column("bm25\nH@5",width=5, justify="center")
    detail_table.add_column("graph\nH@5",width=6, justify="center")
    detail_table.add_column("Top-1 chunk",        max_width=35)

    for r in results:
        is_single = r["source"] == "single_paper"
        tid = r["chunk_id"]

        def hit_str(mode):
            if not is_single:
                return "[dim]—[/dim]"
            ok = r["modes"][mode].get("hits", {}).get("hit@5", False)
            return "[green]✓[/green]" if ok else "[red]✗[/red]"

        top1 = r["modes"]["trimodal"]["top3"][0] if r["modes"]["trimodal"]["top3"] else {}
        top1_str = f"{top1.get('paper','?')[:20]}…\n[dim]{top1.get('section','')[:20]}[/dim]"

        q_type_display = r["type"]
        if r["source"] == "cross_paper":
            q_type_display = f"[cyan]{r['type']}[/cyan] ✦"

        detail_table.add_row(
            str(r["id"]),
            q_type_display,
            r["question"][:38],
            hit_str("trimodal"),
            hit_str("dense"),
            hit_str("bm25"),
            hit_str("graph"),
            top1_str,
        )

    console.print(detail_table)

    # ── Cross-paper questions (manual inspection) ─────────────────────────────
    cross = [r for r in results if r["source"] == "cross_paper"]
    if cross:
        console.print(f"\n[bold]Cross-paper questions — top 3 retrieved chunks:[/bold]")
        for r in cross:
            console.print(f"\n  [bold][{r['id']}] {r['question'][:70]}[/bold]")
            for hit in r["modes"]["trimodal"]["top3"]:
                console.print(
                    f"    [{', '.join(hit['sources'])}] "
                    f"{hit['paper'][:35]:35s} | {hit['section'][:25]:25s} | "
                    f"{hit['text'][:80]}..."
                )

    # ── Save full results ─────────────────────────────────────────────────────
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[dim]Full results saved → {OUT_PATH}[/dim]")


if __name__ == "__main__":
    run_eval()
