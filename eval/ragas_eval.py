"""
Phase 2 — RAGAS evaluation harness.

Runs the full RAG pipeline against the silver QA set and scores with:
  - Faithfulness     : are claims in the answer grounded in retrieved chunks?
  - AnswerRelevancy  : does the answer address the question asked?
  - ContextRecall    : do the retrieved chunks contain the reference answer?
                       (single-paper questions only — needs a reference answer)

API notes (RAGAS 0.4.3):
  - Uses ragas.metrics.collections (new async API), NOT legacy evaluate()
  - Judge LLM: gpt-4o-mini via OpenAI (Anthropic rejected due to RAGAS bug: sends
    both temperature + top_p which Anthropic rejects with 400)
  - Embeddings: text-embedding-3-small (reuses OpenAI key already in use)
  - All scoring is async; we run questions with concurrency=3 to avoid rate limits

Usage:
    python -m eval.ragas_eval
    python -m eval.ragas_eval --n 10       # first 10 questions (dev/cost check)
    python -m eval.ragas_eval --subset single
    python -m eval.ragas_eval --skip-pipeline   # re-score existing pipeline outputs
"""

import argparse
import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv(Path(__file__).parent.parent / ".env", override=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

import openai
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas.metrics.collections import AnswerRelevancy, ContextRecall, Faithfulness

from query.pipeline import RAGPipeline

# ── Paths ─────────────────────────────────────────────────────────────────────

GOLDEN_PATH = Path("docs/golden_set_silver.json")
OUT_PATH    = Path("docs/eval_ragas_round1.json")

# ── Concurrency ───────────────────────────────────────────────────────────────

SCORE_CONCURRENCY = 3   # simultaneous RAGAS scoring calls (rate-limit safety)

console = Console()


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_pipeline_for_all(questions: list[dict], pipeline: RAGPipeline) -> list[dict]:
    """Run the naive RAG pipeline for each question. Returns list of result records."""
    records = []
    for i, qa in enumerate(questions, 1):
        q = qa["question"]
        console.print(f"  [{i:02d}/{len(questions)}] {q[:70]}...")

        result = pipeline.run(q)

        records.append({
            "question":   q,
            "answer":     result.answer,
            "confidence": result.confidence,
            "contexts":   [c.text for c in result.chunks],
            "reference":  qa.get("answer", ""),
            "source":     qa.get("source", "?"),
            "type":       qa.get("type", "?"),
            "chunk_id":   qa.get("chunk_id"),
            "latency_ms": result.latency_ms,
        })

        time.sleep(0.2)  # embedding rate limit

    return records


# ── RAGAS scoring ─────────────────────────────────────────────────────────────

async def score_record(
    record: dict,
    f_metric: Faithfulness,
    ar_metric: AnswerRelevancy,
    cr_metric: ContextRecall,
) -> dict:
    """Score one record with all applicable metrics. Returns dict of scores."""
    scores = {}

    try:
        r = await f_metric.ascore(
            user_input=record["question"],
            response=record["answer"],
            retrieved_contexts=record["contexts"],
        )
        scores["faithfulness"] = round(r.value, 4) if r.value is not None else None
    except Exception as e:
        scores["faithfulness"] = None
        console.print(f"    [red]Faithfulness error: {e}[/red]")

    try:
        r = await ar_metric.ascore(
            user_input=record["question"],
            response=record["answer"],
        )
        scores["answer_relevancy"] = round(r.value, 4) if r.value is not None else None
    except Exception as e:
        scores["answer_relevancy"] = None
        console.print(f"    [red]AnswerRelevancy error: {e}[/red]")

    # ContextRecall only for records that have a reference (single-paper questions)
    if record.get("reference"):
        try:
            r = await cr_metric.ascore(
                user_input=record["question"],
                retrieved_contexts=record["contexts"],
                reference=record["reference"],
            )
            scores["context_recall"] = round(r.value, 4) if r.value is not None else None
        except Exception as e:
            scores["context_recall"] = None
            console.print(f"    [red]ContextRecall error: {e}[/red]")

    return scores


async def score_all(records: list[dict], f_metric, ar_metric, cr_metric) -> list[dict]:
    """Score all records with bounded concurrency."""
    semaphore = asyncio.Semaphore(SCORE_CONCURRENCY)

    async def bounded(i, rec):
        async with semaphore:
            console.print(f"  scoring [{i+1:02d}/{len(records)}] {rec['question'][:60]}...")
            sc = await score_record(rec, f_metric, ar_metric, cr_metric)
            return {**rec, **sc}

    tasks = [bounded(i, rec) for i, rec in enumerate(records)]
    return await asyncio.gather(*tasks)


# ── Aggregate metrics ─────────────────────────────────────────────────────────

def aggregate(scored: list[dict]) -> dict:
    def mean(vals):
        clean = [v for v in vals if v is not None and not math.isnan(v)]
        return round(sum(clean) / len(clean), 4) if clean else None

    return {
        "faithfulness":     mean([r.get("faithfulness")     for r in scored]),
        "answer_relevancy": mean([r.get("answer_relevancy") for r in scored]),
        "context_recall":   mean([r.get("context_recall")   for r in scored
                                   if r.get("context_recall") is not None]),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=["all", "single", "cross"], default="all")
    parser.add_argument("--n",      type=int, default=None)
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="Reuse existing pipeline output (only re-run RAGAS scoring)")
    args = parser.parse_args()

    # ── Load questions ────────────────────────────────────────────────────────
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    if args.subset == "single":
        questions = [q for q in golden if q["source"] == "single_paper"]
    elif args.subset == "cross":
        questions = [q for q in golden if q["source"] == "cross_paper"]
    else:
        questions = golden

    if args.n:
        questions = questions[:args.n]

    console.print(f"\n[bold cyan]PaperMind — Phase 2 RAGAS Baseline[/bold cyan]")
    console.print(f"Questions: {len(questions)} | subset={args.subset} | n={args.n or 'all'}\n")

    # ── Pipeline run ──────────────────────────────────────────────────────────
    if args.skip_pipeline and OUT_PATH.exists():
        console.print("[yellow]Reusing existing pipeline outputs (--skip-pipeline)...[/yellow]")
        with open(OUT_PATH) as f:
            saved = json.load(f)
        records = saved.get("records", [])[:len(questions)]
        console.print(f"  Loaded {len(records)} records from {OUT_PATH}")
    else:
        console.print("[bold]Step 1:[/bold] Running RAG pipeline on all questions...")
        with RAGPipeline(top_k=5) as pipeline:
            records = run_pipeline_for_all(questions, pipeline)

        OUT_PATH.parent.mkdir(exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump({"records": records, "scores": {}}, f, indent=2)
        console.print(f"  [dim]Pipeline outputs saved → {OUT_PATH}[/dim]\n")

    # ── RAGAS setup ───────────────────────────────────────────────────────────
    console.print("[bold]Step 2:[/bold] Scoring with RAGAS (judge=gpt-4o-mini)...")
    oa_async = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    judge_llm = llm_factory("gpt-4o-mini", provider="openai", client=oa_async)
    ragas_emb = OpenAIEmbeddings(model="text-embedding-3-small", client=oa_async)

    f_metric  = Faithfulness(llm=judge_llm)
    ar_metric = AnswerRelevancy(llm=judge_llm, embeddings=ragas_emb)
    cr_metric = ContextRecall(llm=judge_llm)

    scored = asyncio.run(score_all(records, f_metric, ar_metric, cr_metric))
    agg    = aggregate(scored)

    # ── Results table ─────────────────────────────────────────────────────────
    targets = {"faithfulness": 0.85, "answer_relevancy": 0.80, "context_recall": 0.80}

    console.print(f"\n{'='*58}")
    console.print("[bold]RAGAS BASELINE — Phase 2[/bold]")
    console.print(f"{'='*58}")

    tbl = Table(show_header=True, header_style="bold cyan")
    tbl.add_column("Metric",          width=20)
    tbl.add_column("Score",  justify="right", width=8)
    tbl.add_column("Target", justify="right", width=8)
    tbl.add_column("Gap",    justify="right", width=8)

    for metric, score in agg.items():
        if score is None:
            tbl.add_row(metric.replace("_"," ").title(), "—", "—", "—")
            continue
        target  = targets.get(metric)
        gap_str = f"{score - target:+.3f}" if target else "—"
        color   = "green" if (target and score >= target) else "red"
        tbl.add_row(
            metric.replace("_", " ").title(),
            f"[{color}]{score:.3f}[/{color}]",
            f"{target:.2f}" if target else "—",
            f"[{color}]{gap_str}[/{color}]",
        )

    console.print(tbl)
    n_single = sum(1 for r in scored if r["source"] == "single_paper")
    n_cross  = sum(1 for r in scored if r["source"] == "cross_paper")
    console.print(
        f"  n={len(scored)} ({n_single} single, {n_cross} cross) | "
        f"pipeline=claude-sonnet-4-5 | judge=gpt-4o-mini\n"
    )

    # ── Per-question breakdown ────────────────────────────────────────────────
    detail = Table(show_header=True, header_style="bold", show_lines=True)
    detail.add_column("#",      width=3,  justify="right")
    detail.add_column("Src",    width=4)
    detail.add_column("Type",   width=11)
    detail.add_column("Question",           max_width=40)
    detail.add_column("Conf",   width=4,  justify="center")
    detail.add_column("Faith",  width=6,  justify="right")
    detail.add_column("Rel",    width=5,  justify="right")
    detail.add_column("Rec",    width=5,  justify="right")
    detail.add_column("ms",     width=6,  justify="right")

    for i, rec in enumerate(scored, 1):
        faith = rec.get("faithfulness")
        rel   = rec.get("answer_relevancy")
        recall = rec.get("context_recall")
        conf_color = {"high": "green", "medium": "yellow", "low": "red"}.get(
            rec["confidence"], "white"
        )
        src_tag = "[cyan]X[/cyan]" if rec["source"] == "cross_paper" else "S"

        detail.add_row(
            str(i),
            src_tag,
            rec["type"],
            rec["question"][:40],
            f"[{conf_color}]{rec['confidence'][0].upper()}[/{conf_color}]",
            f"{faith:.2f}" if faith is not None else "—",
            f"{rel:.2f}"   if rel   is not None else "—",
            f"{recall:.2f}" if recall is not None else "—",
            f"{rec['latency_ms']:.0f}",
        )

    console.print(detail)

    # ── Save full results ─────────────────────────────────────────────────────
    with open(OUT_PATH, "w") as f:
        json.dump({
            "records":        scored,
            "scores":         agg,
            "n":              len(scored),
            "subset":         args.subset,
            "pipeline_model": "claude-sonnet-4-5",
            "judge_model":    "gpt-4o-mini",
            "ragas_version":  "0.4.3",
        }, f, indent=2)

    console.print(f"\n[dim]Full results → {OUT_PATH}[/dim]")

    # ── One-liner for DESIGN.md ───────────────────────────────────────────────
    console.print("\n[bold]DESIGN.md entry:[/bold]")
    parts = [f"{k.replace('_',' ').title()}: {v:.3f}" for k, v in agg.items() if v]
    console.print(f"  Phase 2 naive RAG baseline | {' | '.join(parts)}")


if __name__ == "__main__":
    main()
