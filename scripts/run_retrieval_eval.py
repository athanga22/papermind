"""
Retrieval evaluation against full_tier / fast_tier.

Metrics by track:

  Chunk-level (requires chunk_ids in test set):
    Hit@k       — was any target chunk in top-k? (binary per question)
    Recall@k    — fraction of target chunks found in top-k (multi-chunk aware)
    MAP@k       — Mean Average Precision; rewards finding all targets and ranking them high.
                  Equals MRR for single-target questions; strictly better for multi-target.

  Paper-level (requires paper_ids in test set):
    Coverage@5  — fraction of expected papers with ≥1 chunk in top-5
    Coverage@10 — fraction of expected papers with ≥1 chunk in top-10
    Full Hit    — % of questions where all expected papers were covered

  Score distribution (adversarial only):
    Avg Top-1 Score — average retrieval score of the top chunk.
                      High score on unanswerable = retriever is hallucinating confidence.

Type assignment:
  factoid, methodology, limitation               → chunk-level
  synthesis                                      → chunk-level (if chunk_ids filled) + paper-level
  false_premise, table_extraction                → chunk-level (once chunk_ids filled) + paper-level
  comparison, contradiction, aggregation         → paper-level
  multi_section                                  → paper-level (single paper; verifies correct paper retrieved)
  adversarial                                    → score distribution only

No LLM calls. Hits Qdrant + BM25 directly.

Usage:
    python scripts/run_retrieval_eval.py
    python scripts/run_retrieval_eval.py --tier full
    python scripts/run_retrieval_eval.py --tier fast --mode dense+bm25
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv(Path(__file__).parent.parent / ".env", override=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

from query.retriever import TrimodalRetriever

EVAL_DIR    = Path(__file__).parent.parent / "eval"
RESULTS_DIR = EVAL_DIR / "results"
K_VALUES    = [1, 3, 5, 10]

# Chunk-level metrics — active when chunk_ids are filled for the question
CHUNK_ID_TYPES = {
    "factoid", "methodology", "limitation",
    "synthesis", "false_premise", "table_extraction",
}

# Paper-level coverage — active when paper_ids are filled
PAPER_COV_TYPES = {
    "comparison", "synthesis", "contradiction", "aggregation",  # cross-paper
    "multi_section", "false_premise", "table_extraction",       # single-paper correctness check
}

MODES = {
    "dense+bm25": dict(use_dense=True,  use_bm25=True,  use_graph=False),
    "dense":      dict(use_dense=True,  use_bm25=False, use_graph=False),
    "bm25":       dict(use_dense=False, use_bm25=True,  use_graph=False),
}

console = Console()


# ── Metric functions ──────────────────────────────────────────────────────────

def hit_at_k(ids: list[str], targets: list[str], k: int) -> bool:
    return any(t in ids[:k] for t in targets)


def recall_at_k(ids: list[str], targets: list[str], k: int) -> float:
    if not targets:
        return 0.0
    return sum(1 for t in targets if t in ids[:k]) / len(targets)


def average_precision_at_k(ids: list[str], targets: list[str], k: int) -> float:
    """
    AP@k = (1 / |R|) * sum_{i=1}^{k} P@i * rel(i)
    where |R| = total relevant, P@i = precision at rank i, rel(i) = 1 if rank i is relevant.
    Equals MRR for single-target questions.
    """
    if not targets:
        return 0.0
    target_set = set(targets)
    hits, score = 0, 0.0
    for i, cid in enumerate(ids[:k], 1):
        if cid in target_set:
            hits += 1
            score += hits / i
    return score / len(target_set)


def paper_coverage_at_k(hits: list, expected_paper_ids: list[str], k: int) -> float:
    """Fraction of expected papers with ≥1 chunk in top-k results."""
    if not expected_paper_ids:
        return 0.0
    retrieved = {h.paper_id for h in hits[:k]}
    return sum(1 for pid in expected_paper_ids if pid in retrieved) / len(expected_paper_ids)


# ── Eval runner ───────────────────────────────────────────────────────────────

def run_eval(tier: str = "fast", primary_mode: str = "dense+bm25") -> None:
    data_path = EVAL_DIR / "data" / f"{tier}_tier.json"
    with open(data_path) as f:
        questions = json.load(f)

    ts       = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = RESULTS_DIR / f"{ts}_retrieval_{tier}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    n_adversarial = sum(1 for q in questions if q.get("type") == "adversarial")
    console.print(f"\n[bold cyan]PaperMind — Retrieval Eval[/bold cyan]")
    console.print(
        f"Tier: [bold]{tier}[/bold] | "
        f"Questions: {len(questions)} ({n_adversarial} adversarial — score-dist only)"
    )
    console.print(f"Modes: {', '.join(MODES)} | K: {K_VALUES}\n")

    results = []

    with TrimodalRetriever() as retriever:
        for i, qa in enumerate(questions, 1):
            q           = qa["question"]
            qtype       = qa.get("type", "?")
            target_cids = [c for c in (qa.get("chunk_id") or []) if c]
            paper_ids   = qa.get("paper_ids") or []

            console.print(f"  [{i:02d}/{len(questions)}] [{qtype}] {q[:70]}...")

            mode_results: dict = {}
            for mode_name, kwargs in MODES.items():
                hits     = retriever.retrieve(q, top_k=10, **kwargs)
                ret_cids = [h.chunk_id for h in hits]

                entry: dict = {
                    "top3": [
                        {
                            "chunk_id": h.chunk_id,
                            "paper_id": h.paper_id,
                            "paper":    h.paper_title[:50],
                            "section":  h.section,
                            "text":     h.text[:150],
                            "score":    round(h.score, 4),
                            "sources":  h.sources,
                        }
                        for h in hits[:3]
                    ],
                }

                if qtype == "adversarial":
                    entry["top1_score"] = round(hits[0].score, 4) if hits else 0.0

                elif target_cids and qtype in CHUNK_ID_TYPES:
                    entry["hits"]   = {f"hit@{k}":    hit_at_k(ret_cids, target_cids, k)   for k in K_VALUES}
                    entry["recall"] = {f"recall@{k}": recall_at_k(ret_cids, target_cids, k) for k in K_VALUES}
                    entry["map"]    = {f"map@{k}":    average_precision_at_k(ret_cids, target_cids, k) for k in K_VALUES}

                if paper_ids and qtype in PAPER_COV_TYPES:
                    entry["paper_coverage"] = {
                        "cov@5":  paper_coverage_at_k(hits, paper_ids, 5),
                        "cov@10": paper_coverage_at_k(hits, paper_ids, 10),
                    }

                mode_results[mode_name] = entry

            results.append({
                "id":        i,
                "question":  q,
                "type":      qtype,
                "source":    qa.get("source", "?"),
                "chunk_ids": target_cids,
                "paper_ids": paper_ids,
                "modes":     mode_results,
            })
            time.sleep(0.1)

    _print_results(results, primary_mode)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[dim]Results → {out_path}[/dim]")


# ── Display ───────────────────────────────────────────────────────────────────

def _mean(vals: list[float]) -> float | None:
    clean = [v for v in vals if v is not None]
    return sum(clean) / len(clean) if clean else None


def _print_results(results: list[dict], primary_mode: str) -> None:
    hit_q  = [r for r in results if r["modes"][primary_mode].get("hits")]
    cov_q  = [r for r in results if r["modes"][primary_mode].get("paper_coverage") is not None]
    adv_q  = [r for r in results if r["type"] == "adversarial"]

    # ── Chunk-level summary ───────────────────────────────────────────────────
    console.print(f"\n{'='*65}")
    console.print(f"[bold]Chunk-level metrics (n={len(hit_q)}, types with chunk_ids)[/bold]")
    console.print(f"{'='*65}")

    tbl = Table(show_header=True, header_style="bold cyan")
    tbl.add_column("Mode",    width=12)
    tbl.add_column("Hit@5",   justify="right", width=7)
    tbl.add_column("Hit@10",  justify="right", width=7)
    tbl.add_column("Rec@5",   justify="right", width=7)
    tbl.add_column("Rec@10",  justify="right", width=7)
    tbl.add_column("MAP@5",   justify="right", width=7)
    tbl.add_column("MAP@10",  justify="right", width=7)

    for mode in MODES:
        mq = [r for r in results if r["modes"][mode].get("hits")]
        if not mq:
            continue
        n = len(mq)
        h, rec, ap = defaultdict(int), defaultdict(float), defaultdict(float)
        for r in mq:
            m = r["modes"][mode]
            for k in K_VALUES:
                if m["hits"].get(f"hit@{k}"):    h[k]   += 1
                rec[k] += m["recall"].get(f"recall@{k}", 0.0)
                ap[k]  += m["map"].get(f"map@{k}",    0.0)
        bold = mode == primary_mode
        s    = "bold green" if bold else ""
        fmt  = (lambda _s: lambda v: f"[{_s}]{v}[/{_s}]" if _s else str(v))(s)
        tbl.add_row(
            fmt(mode),
            fmt(f"{h[5]/n:.0%}"),  fmt(f"{h[10]/n:.0%}"),
            fmt(f"{rec[5]/n:.2f}"), fmt(f"{rec[10]/n:.2f}"),
            fmt(f"{ap[5]/n:.3f}"),  fmt(f"{ap[10]/n:.3f}"),
        )
    console.print(tbl)

    # per-type breakdown
    console.print(f"\n[bold]Chunk-level by type ({primary_mode})[/bold]")
    type_tbl = Table(show_header=True, header_style="bold")
    type_tbl.add_column("Type",   width=18)
    type_tbl.add_column("Hit@5",  justify="right", width=7)
    type_tbl.add_column("Rec@5",  justify="right", width=7)
    type_tbl.add_column("MAP@5",  justify="right", width=7)
    type_tbl.add_column("n",      justify="right", width=4)

    by_type: dict = defaultdict(list)
    for r in hit_q:
        by_type[r["type"]].append(r)
    for typ, rows in sorted(by_type.items()):
        n    = len(rows)
        h5   = sum(1 for r in rows if r["modes"][primary_mode]["hits"].get("hit@5"))
        rec5 = sum(r["modes"][primary_mode]["recall"].get("recall@5", 0.0) for r in rows) / n
        ap5  = sum(r["modes"][primary_mode]["map"].get("map@5",    0.0) for r in rows) / n
        type_tbl.add_row(typ, f"{h5/n:.0%}", f"{rec5:.2f}", f"{ap5:.3f}", str(n))
    console.print(type_tbl)

    # ── Paper coverage ────────────────────────────────────────────────────────
    if cov_q:
        console.print(f"\n[bold]Paper coverage (n={len(cov_q)})[/bold]")
        cov_tbl = Table(show_header=True, header_style="bold magenta")
        cov_tbl.add_column("Type",     width=18)
        cov_tbl.add_column("Cov@5",    justify="right", width=8)
        cov_tbl.add_column("Cov@10",   justify="right", width=8)
        cov_tbl.add_column("Full@5",   justify="right", width=8)
        cov_tbl.add_column("n",        justify="right", width=4)

        by_cov: dict = defaultdict(list)
        for r in cov_q:
            by_cov[r["type"]].append(r["modes"][primary_mode]["paper_coverage"])

        for typ, covs in sorted(by_cov.items()):
            n      = len(covs)
            avg5   = _mean([c["cov@5"]  for c in covs])
            avg10  = _mean([c["cov@10"] for c in covs])
            full5  = sum(1 for c in covs if c["cov@5"] >= 1.0) / n
            cov_tbl.add_row(typ, f"{avg5:.2f}", f"{avg10:.2f}", f"{full5:.0%}", str(n))

        all_cov5  = _mean([r["modes"][primary_mode]["paper_coverage"]["cov@5"]  for r in cov_q])
        all_cov10 = _mean([r["modes"][primary_mode]["paper_coverage"]["cov@10"] for r in cov_q])
        cov_tbl.add_row(
            "[bold]overall[/bold]",
            f"[bold]{all_cov5:.2f}[/bold]", f"[bold]{all_cov10:.2f}[/bold]",
            "—", str(len(cov_q)),
        )
        console.print(cov_tbl)

    # ── Adversarial score distribution ────────────────────────────────────────
    if adv_q:
        scores = [
            r["modes"][primary_mode].get("top1_score", 0.0)
            for r in adv_q
            if r["modes"][primary_mode].get("top1_score") is not None
        ]
        if scores:
            avg  = sum(scores) / len(scores)
            high = sum(1 for s in scores if s > 0.5)
            console.print(f"\n[bold]Adversarial score distribution (n={len(scores)})[/bold]")
            adv_tbl = Table(show_header=True, header_style="bold red")
            adv_tbl.add_column("Metric",              width=30)
            adv_tbl.add_column("Value", justify="right", width=10)
            adv_tbl.add_column("Note",                width=40)
            adv_tbl.add_row(
                "Avg top-1 retrieval score", f"{avg:.3f}",
                "[green]low is good[/green] — retriever is uncertain",
            )
            adv_tbl.add_row(
                "High-confidence retrievals (>0.5)",
                f"{high}/{len(scores)}",
                "[red]bad[/red] — confident hits on unanswerable Qs" if high > 0 else "[green]none[/green]",
            )
            console.print(adv_tbl)

    # ── Per-question detail ───────────────────────────────────────────────────
    detail = Table(show_header=True, header_style="bold", show_lines=True)
    detail.add_column("#",       width=3,  justify="right")
    detail.add_column("Type",    width=16)
    detail.add_column("Question",            max_width=38)
    detail.add_column("H@5",     width=4,   justify="center")
    detail.add_column("R@5",     width=5,   justify="right")
    detail.add_column("Cov@5",   width=6,   justify="right")
    detail.add_column("Top-1",               max_width=28)

    for r in results:
        m    = r["modes"][primary_mode]
        hits = m.get("hits", {})
        cov  = m.get("paper_coverage", {})

        h5       = hits.get("hit@5")
        rec5     = m.get("recall", {}).get("recall@5")
        cov5     = cov.get("cov@5") if cov else None
        top1     = m["top3"][0] if m["top3"] else {}

        if r["type"] == "adversarial":
            h5_str = "[dim]adv[/dim]"
        elif hits:
            h5_str = "[green]✓[/green]" if h5 else "[red]✗[/red]"
        else:
            h5_str = "[dim]—[/dim]"

        rec5_str = f"{rec5:.2f}" if rec5 is not None else "[dim]—[/dim]"
        cov5_str = f"{cov5:.2f}" if cov5 is not None else "[dim]—[/dim]"
        top1_str = f"{top1.get('paper','?')[:18]}\n[dim]{top1.get('section','')[:16]}[/dim]"
        src_color = "cyan" if r["source"] == "cross_paper" else "white"

        detail.add_row(
            str(r["id"]),
            f"[{src_color}]{r['type'][:16]}[/{src_color}]",
            r["question"][:38],
            h5_str, rec5_str, cov5_str, top1_str,
        )

    console.print(detail)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=["fast", "full"], default="fast")
    parser.add_argument("--mode", default="dense+bm25", choices=list(MODES))
    args = parser.parse_args()
    run_eval(tier=args.tier, primary_mode=args.mode)
