"""
PaperMind RAGAS evaluation harness — v2.

Two tiers:
  fast  — 14 questions (7 single + 7 cross-paper reviewed). Run on every change.
  full  — 34 questions (27 single + 7 cross-paper reviewed). Run before "release".

Results are written to eval/results/ with a timestamp — never overwritten.

Usage:
    # Fast tier (default) — agent pipeline:
    python -u -m eval.run_eval

    # Full tier:
    python -u -m eval.run_eval --tier full

    # Dev / cost check:
    python -u -m eval.run_eval --n 5

    # Re-score existing pipeline output (skip re-running the agent):
    python -u -m eval.run_eval --skip-pipeline --checkpoint eval/results/2026-04-15_...json

    # Start fresh (ignore any checkpoint):
    python -u -m eval.run_eval --fresh

API notes:
  - Judge LLM : gpt-4o-mini (OpenAI) — Anthropic rejected by RAGAS due to temp+top_p bug
  - Embeddings: text-embedding-3-small (OpenAI)
  - Scoring is async with concurrency=3 to avoid rate limits
  - Pipeline outputs checkpointed after each question — safe to kill and resume
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv(Path(__file__).parent.parent / ".env", override=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

import openai
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from eval.thresholds import THRESHOLDS
from query.agent import run_agent
from query.tracing import shutdown as langfuse_shutdown

# ── Paths ─────────────────────────────────────────────────────────────────────

EVAL_DIR     = Path(__file__).parent
DATA_DIR     = EVAL_DIR / "data"
RESULTS_DIR  = EVAL_DIR / "results"

TIER_PATHS = {
    "fast": DATA_DIR / "fast_tier.json",
    "full": DATA_DIR / "full_tier.json",
}

SCORE_CONCURRENCY = 3

console = Console(force_terminal=True, record=True)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _load_checkpoint(path: Path) -> dict:
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {"records": [], "scores": {}}


def _save_checkpoint(path: Path, records: list[dict], scores: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"records": records, "scores": scores or {}}, f, indent=2)


# ── Agent runner ──────────────────────────────────────────────────────────────

def run_agent_for_all(
    questions: list[dict],
    existing: list[dict],
    checkpoint_path: Path,
) -> list[dict]:
    done    = {r["question"] for r in existing}
    records = list(existing)
    pending = [qa for qa in questions if qa["question"] not in done]

    if not pending:
        console.print("  [green]All pipeline records exist — skipping Step 1.[/green]")
        return records

    if done:
        console.print(
            f"  [yellow]Resuming: {len(done)} done, {len(pending)} remaining.[/yellow]"
        )

    total = len(questions)
    for qa in pending:
        i   = len(records) + 1
        q   = qa["question"]
        console.print(f"  [{i:02d}/{total}] {q[:72]}...")

        t0          = time.time()
        state       = run_agent(q, use_cache=False)
        latency_ms  = (time.time() - t0) * 1000

        chunks = state.get("retrieved_chunks") or []
        seen: set[str] = set()
        contexts: list[str] = []
        for c in chunks:
            cid = str(c.get("chunk_id", ""))
            if cid not in seen:
                seen.add(cid)
                contexts.append(str(c.get("text", "")))

        node_lats = dict(state.get("stage_latencies") or {})
        node_lats.update({
            "replan_count":  state.get("replan_count", 0),
            "n_sub_queries": len(state.get("sub_queries") or []),
            "n_chunks_raw":  len(chunks),
        })

        records.append({
            "question":        q,
            "answer":          state.get("synthesis") or "(no answer)",
            "confidence":      (
                "high"   if (state.get("confidence_score") or 0) >= 0.8 else
                "medium" if (state.get("confidence_score") or 0) >= 0.5 else
                "low"
            ),
            "contexts":        contexts,
            "reference":       qa.get("answer", ""),
            "source":          qa.get("source", "?"),
            "type":            qa.get("type", "?"),
            "chunk_id":        qa.get("chunk_id"),
            "latency_ms":      latency_ms,
            "stage_latencies": node_lats,
        })
        _save_checkpoint(checkpoint_path, records)

    return records


# ── RAGAS scoring ─────────────────────────────────────────────────────────────

async def _score_one(
    record: dict,
    f_metric, ar_metric, cr_metric, cp_metric,
) -> dict:
    scores: dict = {}

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

        try:
            r = await cp_metric.ascore(
                user_input=record["question"],
                retrieved_contexts=record["contexts"],
                reference=record["reference"],
            )
            scores["context_precision"] = round(r.value, 4) if r.value is not None else None
        except Exception as e:
            scores["context_precision"] = None
            console.print(f"    [red]ContextPrecision error: {e}[/red]")

    return scores


async def score_all(records, f_metric, ar_metric, cr_metric, cp_metric, checkpoint_path) -> list[dict]:
    sem     = asyncio.Semaphore(SCORE_CONCURRENCY)
    results = list(records)

    async def bounded(i: int, rec: dict) -> None:
        if rec.get("faithfulness") is not None:
            console.print(f"  scoring [{i+1:02d}/{len(results)}] already scored — skip")
            return
        async with sem:
            console.print(f"  scoring [{i+1:02d}/{len(results)}] {rec['question'][:62]}...")
            sc = await _score_one(rec, f_metric, ar_metric, cr_metric, cp_metric)
            results[i] = {**rec, **sc}
            _save_checkpoint(checkpoint_path, results)

    await asyncio.gather(*[bounded(i, r) for i, r in enumerate(results)])
    return results


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(scored: list[dict]) -> dict:
    def mean(vals):
        clean = [v for v in vals if v is not None and not math.isnan(v)]
        return round(sum(clean) / len(clean), 4) if clean else None

    return {
        "faithfulness":      mean([r.get("faithfulness")      for r in scored]),
        "answer_relevancy":  mean([r.get("answer_relevancy")  for r in scored]),
        "context_recall":    mean([r.get("context_recall")    for r in scored
                                   if r.get("context_recall")    is not None]),
        "context_precision": mean([r.get("context_precision") for r in scored
                                   if r.get("context_precision") is not None]),
    }


def latency_stats(scored: list[dict]) -> dict:
    def pct(vals, p):
        clean = sorted(v for v in vals if v is not None and v > 0)
        if not clean: return None
        return round(clean[max(0, int(len(clean) * p / 100) - 1)], 1)

    totals = [r.get("latency_ms") for r in scored]
    stage_keys: set[str] = set()
    for r in scored:
        stage_keys.update((r.get("stage_latencies") or {}).keys())

    stats: dict = {"total_p50_ms": pct(totals, 50), "total_p95_ms": pct(totals, 95)}
    for key in sorted(stage_keys):
        vals = [r.get("stage_latencies", {}).get(key) for r in scored]
        stats[f"{key}_p50"] = pct(vals, 50)
        stats[f"{key}_p95"] = pct(vals, 95)
    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier",           choices=["fast", "full"], default="fast",
                        help="fast=14q (run often) | full=34q (pre-release)")
    parser.add_argument("--n",              type=int, default=None,
                        help="Limit to first N questions (dev/cost check)")
    parser.add_argument("--fresh",          action="store_true",
                        help="Ignore checkpoint, start from scratch")
    parser.add_argument("--skip-pipeline",  action="store_true",
                        help="Reuse existing pipeline output, only re-run scoring")
    parser.add_argument("--checkpoint",     type=Path, default=None,
                        help="Explicit checkpoint path to resume from")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Output path: timestamped, never overwritten
    ts       = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = args.checkpoint or (RESULTS_DIR / f"{ts}_{args.tier}.json")
    log_path = out_path.with_suffix(".log")

    # ── Load questions ────────────────────────────────────────────────────────
    with open(TIER_PATHS[args.tier]) as f:
        questions = json.load(f)

    if args.n:
        questions = questions[:args.n]

    n_single = sum(1 for q in questions if q["source"] == "single_paper")
    n_cross  = sum(1 for q in questions if q["source"] == "cross_paper")

    synthesis_model = os.getenv("PAPERMIND_SYNTHESIS_MODEL", "claude-sonnet-4-5-20250929")
    _short = synthesis_model.replace("claude-", "").split("-20")[0]

    console.print(f"\n[bold cyan]PaperMind — RAGAS Eval[/bold cyan]")
    console.print(
        f"Tier: [bold]{args.tier}[/bold] | "
        f"Questions: {len(questions)} ({n_single} single, {n_cross} cross) | "
        f"synthesis: {_short}"
    )
    console.print(f"[dim]Output → {out_path}[/dim]\n")

    # ── Checkpoint ────────────────────────────────────────────────────────────
    checkpoint     = {} if args.fresh else _load_checkpoint(out_path)
    existing_recs  = checkpoint.get("records", [])
    if existing_recs and not args.fresh:
        console.print(f"[dim]Checkpoint: {len(existing_recs)} records in {out_path}[/dim]")

    # ── Pipeline run ──────────────────────────────────────────────────────────
    if args.skip_pipeline and existing_recs:
        console.print("[yellow]--skip-pipeline: reusing existing outputs[/yellow]")
        records = existing_recs[:len(questions)]
    else:
        console.print("[bold]Step 1:[/bold] Running agent pipeline...")
        records = run_agent_for_all(questions, existing_recs, out_path)
        console.print(f"  [dim]Saved → {out_path}[/dim]\n")

    # ── RAGAS scoring ─────────────────────────────────────────────────────────
    console.print("[bold]Step 2:[/bold] Scoring with RAGAS (judge=gpt-4o-mini)...")
    oa        = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    judge_llm = llm_factory("gpt-4o-mini", provider="openai", client=oa, max_tokens=16000)
    ragas_emb = OpenAIEmbeddings(model="text-embedding-3-small", client=oa)

    f_metric  = Faithfulness(llm=judge_llm)
    ar_metric = AnswerRelevancy(llm=judge_llm, embeddings=ragas_emb)
    cr_metric = ContextRecall(llm=judge_llm)
    cp_metric = ContextPrecision(llm=judge_llm)

    scored = asyncio.run(score_all(records, f_metric, ar_metric, cr_metric, cp_metric, out_path))
    agg    = aggregate(scored)
    lats   = latency_stats(scored)

    # ── Results table ─────────────────────────────────────────────────────────
    console.print(f"\n{'='*60}")
    console.print(f"[bold]RAGAS RESULTS — {args.tier.upper()} TIER[/bold]")
    console.print(f"{'='*60}")

    tbl = Table(show_header=True, header_style="bold cyan")
    tbl.add_column("Metric",         width=22)
    tbl.add_column("Score",  justify="right", width=8)
    tbl.add_column("Target", justify="right", width=8)
    tbl.add_column("Gap",    justify="right", width=8)
    tbl.add_column("",       width=2)

    all_pass = True
    for metric, score in agg.items():
        target = THRESHOLDS.get(metric)
        if score is None:
            tbl.add_row(metric.replace("_", " ").title(), "—", "—", "—", "")
            continue
        passed  = target is None or score >= target
        all_pass = all_pass and passed
        color   = "green" if passed else "red"
        gap_str = f"{score - target:+.3f}" if target else "—"
        status  = "✅" if passed else "❌"
        tbl.add_row(
            metric.replace("_", " ").title(),
            f"[{color}]{score:.3f}[/{color}]",
            f"{target:.2f}" if target else "—",
            f"[{color}]{gap_str}[/{color}]",
            status,
        )

    console.print(tbl)
    gate_color = "green" if all_pass else "red"
    gate_label = "ALL TARGETS MET ✅" if all_pass else "REGRESSION DETECTED ❌"
    console.print(f"  [{gate_color}]{gate_label}[/{gate_color}]")
    console.print(
        f"  n={len(scored)} ({n_single} single, {n_cross} cross) | "
        f"judge=gpt-4o-mini | synthesis={_short}\n"
    )

    # ── Latency table ─────────────────────────────────────────────────────────
    console.print("[bold]Latency (ms)[/bold]")
    lat_tbl = Table(show_header=True, header_style="bold")
    lat_tbl.add_column("Stage",  width=18)
    lat_tbl.add_column("p50", justify="right", width=8)
    lat_tbl.add_column("p95", justify="right", width=8)

    lat_tbl.add_row("Total", str(lats.get("total_p50_ms", "—")), str(lats.get("total_p95_ms", "—")))
    for key in ("classifier_ms", "planner_ms", "retrieve_ms",
                "rerank_ms", "gate_ms", "replan_ms", "synthesis_ms"):
        p50 = lats.get(f"{key}_p50")
        p95 = lats.get(f"{key}_p95")
        if p50 is not None:
            lat_tbl.add_row(key.replace("_ms", ""), str(p50), str(p95))

    console.print(lat_tbl)

    # ── Per-question breakdown ────────────────────────────────────────────────
    detail = Table(show_header=True, header_style="bold", show_lines=True)
    detail.add_column("#",       width=3,  justify="right")
    detail.add_column("Src",     width=4)
    detail.add_column("Type",    width=11)
    detail.add_column("Question",            max_width=42)
    detail.add_column("Conf",    width=4,   justify="center")
    detail.add_column("Faith",   width=6,   justify="right")
    detail.add_column("Rel",     width=5,   justify="right")
    detail.add_column("Rec",     width=5,   justify="right")
    detail.add_column("Prec",    width=5,   justify="right")
    detail.add_column("ms",      width=6,   justify="right")

    for i, rec in enumerate(scored, 1):
        faith  = rec.get("faithfulness")
        rel    = rec.get("answer_relevancy")
        recall = rec.get("context_recall")
        prec   = rec.get("context_precision")
        conf_color = {"high": "green", "medium": "yellow", "low": "red"}.get(rec["confidence"], "white")
        src_tag    = "[cyan]X[/cyan]" if rec["source"] == "cross_paper" else "S"
        detail.add_row(
            str(i), src_tag, rec["type"], rec["question"][:42],
            f"[{conf_color}]{rec['confidence'][0].upper()}[/{conf_color}]",
            f"{faith:.2f}"  if faith  is not None else "—",
            f"{rel:.2f}"    if rel    is not None else "—",
            f"{recall:.2f}" if recall is not None else "—",
            f"{prec:.2f}"   if prec   is not None else "—",
            f"{rec['latency_ms']:.0f}",
        )

    console.print(detail)

    # ── Save final results ────────────────────────────────────────────────────
    _save_checkpoint(out_path, scored, {
        "aggregates":     agg,
        "latency_stats":  lats,
        "n":              len(scored),
        "tier":           args.tier,
        "all_pass":       all_pass,
        "synthesis_model": synthesis_model,
        "judge_model":    "gpt-4o-mini",
        "ragas_version":  "0.4.3",
        "timestamp":      ts,
    })

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write(console.export_text(clear=False))
    console.print(f"\n[dim]Results → {out_path}[/dim]")
    console.print(f"[dim]Log     → {log_path}[/dim]")

    langfuse_shutdown()

    # ── DESIGN.md one-liner ───────────────────────────────────────────────────
    parts = [f"{k.replace('_',' ').title()}: {v:.3f}" for k, v in agg.items() if v]
    console.print(f"\n[bold]DESIGN.md entry:[/bold]")
    console.print(f"  {_short} | {' | '.join(parts)}")
    console.print(
        f"  Latency p50={lats.get('total_p50_ms')}ms p95={lats.get('total_p95_ms')}ms | "
        f"synthesis p50={lats.get('synthesis_ms_p50')}ms"
    )


if __name__ == "__main__":
    main()
