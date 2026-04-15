"""
Phase 2/3 — RAGAS evaluation harness.

Runs the full RAG pipeline against the silver QA set and scores with:
  - Faithfulness     : are claims in the answer grounded in retrieved chunks?
  - AnswerRelevancy  : does the answer address the question asked?
  - ContextRecall    : do the retrieved chunks contain the reference answer?
                       (single-paper questions only — needs a reference answer)
  - ContextPrecision : are the retrieved chunks actually relevant?

API notes (RAGAS 0.4.3):
  - Uses ragas.metrics.collections (new async API), NOT legacy evaluate()
  - Judge LLM: gpt-4o-mini via OpenAI (Anthropic rejected due to RAGAS bug: sends
    both temperature + top_p which Anthropic rejects with 400)
  - Embeddings: text-embedding-3-small (reuses OpenAI key already in use)
  - All scoring is async; we run questions with concurrency=3 to avoid rate limits

Crash safety:
  - Pipeline results are checkpointed to OUT_PATH after EACH question.
  - On restart, already-completed questions are skipped automatically.
  - RAGAS scoring results are also checkpointed per-question.
  - Full run log is saved to OUT_LOG alongside the JSON.

Usage:
    # Phase 2 baseline (hybrid, no rerank):
    python -u -m eval.ragas_eval

    # Phase 3 Step 2 (hybrid + Cohere rerank):
    python -u -m eval.ragas_eval --rerank

    # Phase 3 Step 3 (hybrid + graph only):
    python -u -m eval.ragas_eval --graph

    # Phase 3 Step 3 full (hybrid + graph + rerank):
    python -u -m eval.ragas_eval --graph --rerank

    # Dev/cost check:
    python -u -m eval.ragas_eval --n 10
    python -u -m eval.ragas_eval --rerank --n 10

    # Re-score existing pipeline output (skip re-running pipeline):
    python -u -m eval.ragas_eval --skip-pipeline

    # Start fresh (ignore checkpoint):
    python -u -m eval.ragas_eval --fresh
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
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from query.agent import run_agent
from query.pipeline import RAGPipeline
from query.tracing import shutdown as langfuse_shutdown

# ── Paths ─────────────────────────────────────────────────────────────────────

SINGLE_PATH = Path("test set/single_paper.json")
CROSS_PATH  = Path("test set/cross_paper.json")

# ── Concurrency ───────────────────────────────────────────────────────────────

SCORE_CONCURRENCY = 3   # simultaneous RAGAS scoring calls (rate-limit safety)

# force_terminal=True: print even when stdout is not a real TTY (e.g. CI, subprocess).
# record=True: captures everything for OUT_LOG.
console = Console(force_terminal=True, record=True)


# ── Output path helpers ───────────────────────────────────────────────────────

def _output_paths(args) -> tuple[Path, Path]:
    """Derive output JSON + log paths from active flags."""
    if args.out:
        p = args.out
        return p, p.with_suffix(".log")

    parts = ["eval_ragas"]
    if getattr(args, "agent", False):
        parts.append("agent")
    elif args.rerank and args.graph:
        parts.append("step3_full")
    elif args.rerank:
        parts.append("step2_rerank")
    elif args.graph:
        parts.append("step3_graph")
    else:
        parts.append("round1")

    stem = "_".join(parts)
    base = Path("docs") / stem
    return base.with_suffix(".json"), base.with_suffix(".log")


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _load_checkpoint(path: Path) -> dict:
    """Load existing checkpoint if present. Returns {'records': [], 'scores': {}}."""
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {"records": [], "scores": {}}


def _save_checkpoint(path: Path, records: list[dict], scores: dict | None = None) -> None:
    """Persist current records (and optional scores) to path immediately."""
    path.parent.mkdir(exist_ok=True)
    payload = {"records": records, "scores": scores or {}}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_pipeline_for_all(
    questions: list[dict],
    pipeline: RAGPipeline,
    existing: list[dict],
    out_path: Path,
    inter_question_sleep: float = 0.2,
) -> list[dict]:
    """
    Run the RAG pipeline for each question.

    Skips questions that already have a record in `existing` (matched by question
    text). Checkpoints to disk after every question — safe to kill and resume.
    """
    done = {r["question"] for r in existing}
    records = list(existing)

    pending = [qa for qa in questions if qa["question"] not in done]
    if not pending:
        console.print("  [green]All pipeline records already exist — skipping Step 1.[/green]")
        return records

    if done:
        console.print(
            f"  [yellow]Resuming: {len(done)} already done, "
            f"{len(pending)} remaining.[/yellow]"
        )

    total = len(questions)
    for qa in pending:
        i = len(records) + 1
        q = qa["question"]
        console.print(f"  [{i:02d}/{total}] {q[:70]}...")

        result = pipeline.run(q)

        record = {
            "question":        q,
            "answer":          result.answer,
            "confidence":      result.confidence,
            "contexts":        [c.text for c in result.chunks],
            "reference":       qa.get("answer", ""),
            "source":          qa.get("source", "?"),
            "type":            qa.get("type", "?"),
            "chunk_id":        qa.get("chunk_id"),
            "latency_ms":      result.latency_ms,
            "stage_latencies": result.stage_latencies,
        }
        records.append(record)
        _save_checkpoint(out_path, records)

        time.sleep(inter_question_sleep)

    return records


# ── Agent pipeline runner ─────────────────────────────────────────────────────

def run_agent_for_all(
    questions: list[dict],
    existing: list[dict],
    out_path: Path,
) -> list[dict]:
    """
    Run the LangGraph agentic pipeline for each question.

    Mirrors `run_pipeline_for_all` but uses `run_agent()` instead of
    `RAGPipeline.run()`. Cache is disabled during eval to get true metrics.
    """
    done = {r["question"] for r in existing}
    records = list(existing)

    pending = [qa for qa in questions if qa["question"] not in done]
    if not pending:
        console.print("  [green]All agent records already exist — skipping Step 1.[/green]")
        return records

    if done:
        console.print(
            f"  [yellow]Resuming: {len(done)} already done, "
            f"{len(pending)} remaining.[/yellow]"
        )

    total = len(questions)
    for qa in pending:
        i = len(records) + 1
        q = qa["question"]
        console.print(f"  [{i:02d}/{total}] {q[:70]}...")

        t0 = time.time()
        state = run_agent(q, use_cache=False)
        latency_ms = (time.time() - t0) * 1000

        # Extract contexts from retrieved_chunks (same format eval scorer expects)
        chunks = state.get("retrieved_chunks") or []
        # Dedup by chunk_id for contexts
        seen_cids: set[str] = set()
        contexts: list[str] = []
        for c in chunks:
            cid = str(c.get("chunk_id", ""))
            if cid not in seen_cids:
                seen_cids.add(cid)
                contexts.append(str(c.get("text", "")))

        # Merge real per-node timings with pipeline metadata
        node_lats = dict(state.get("stage_latencies") or {})
        node_lats.update({
            "replan_count":  state.get("replan_count", 0),
            "n_sub_queries": len(state.get("sub_queries") or []),
            "n_chunks_raw":  len(chunks),
        })

        record = {
            "question":        q,
            "answer":          state.get("synthesis") or "(no answer)",
            "confidence":      "high" if (state.get("confidence_score") or 0) >= 0.8
                               else "medium" if (state.get("confidence_score") or 0) >= 0.5
                               else "low",
            "contexts":        contexts,
            "reference":       qa.get("answer", ""),
            "source":          qa.get("source", "?"),
            "type":            qa.get("type", "?"),
            "chunk_id":        qa.get("chunk_id"),
            "latency_ms":      latency_ms,
            "stage_latencies": node_lats,
        }
        records.append(record)
        _save_checkpoint(out_path, records)

    return records


# ── RAGAS scoring ─────────────────────────────────────────────────────────────

async def score_record(
    record: dict,
    f_metric:  Faithfulness,
    ar_metric: AnswerRelevancy,
    cr_metric: ContextRecall,
    cp_metric: ContextPrecision,
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

    # ContextRecall + ContextPrecision only for records that have a reference answer
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


async def score_all(
    records: list[dict],
    f_metric,
    ar_metric,
    cr_metric,
    cp_metric,
    out_path: Path,
) -> list[dict]:
    """
    Score all records with bounded concurrency.

    Skips records that already have a faithfulness score (resume support).
    Checkpoints to disk after each scored record.
    """
    semaphore = asyncio.Semaphore(SCORE_CONCURRENCY)
    results = list(records)

    async def bounded(i: int, rec: dict) -> None:
        if rec.get("faithfulness") is not None:
            console.print(f"  scoring [{i+1:02d}/{len(results)}] already scored — skip")
            return

        async with semaphore:
            console.print(
                f"  scoring [{i+1:02d}/{len(results)}] {rec['question'][:60]}..."
            )
            sc = await score_record(rec, f_metric, ar_metric, cr_metric, cp_metric)
            results[i] = {**rec, **sc}
            _save_checkpoint(out_path, results)

    await asyncio.gather(*[bounded(i, rec) for i, rec in enumerate(results)])
    return results


# ── Aggregate metrics ─────────────────────────────────────────────────────────

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


# ── Latency percentiles ───────────────────────────────────────────────────────

def latency_stats(scored: list[dict]) -> dict:
    """Compute p50/p95 for total latency and each retrieval stage."""
    def percentile(vals: list, p: int) -> float | None:
        clean = sorted(v for v in vals if v is not None and v > 0)
        if not clean:
            return None
        idx = max(0, int(len(clean) * p / 100) - 1)
        return round(clean[idx], 1)

    totals = [r.get("latency_ms") for r in scored]

    stage_keys: set[str] = set()
    for r in scored:
        stage_keys.update((r.get("stage_latencies") or {}).keys())

    stats: dict = {
        "total_p50_ms": percentile(totals, 50),
        "total_p95_ms": percentile(totals, 95),
    }
    for key in sorted(stage_keys):
        vals = [r.get("stage_latencies", {}).get(key) for r in scored]
        stats[f"{key}_p50"] = percentile(vals, 50)
        stats[f"{key}_p95"] = percentile(vals, 95)

    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset",             choices=["all", "single", "cross"], default="all")
    parser.add_argument("--n",                  type=int, default=None)
    parser.add_argument("--rerank",             action="store_true",
                        help="Enable Cohere cross-encoder reranking (Phase 3 Step 2)")
    parser.add_argument("--graph",              action="store_true",
                        help="Enable Neo4j graph traversal (Phase 3 Step 3)")
    parser.add_argument("--skip-pipeline",      action="store_true",
                        help="Reuse existing pipeline output (only re-run RAGAS scoring)")
    parser.add_argument("--agent",              action="store_true",
                        help="Use LangGraph agentic pipeline (Phase 4) instead of naive RAG")
    parser.add_argument("--fresh",              action="store_true",
                        help="Ignore any existing checkpoint and start from scratch")
    parser.add_argument("--include-unreviewed", action="store_true",
                        help="Include cross-paper items flagged reviewed=false")
    parser.add_argument("--out",                type=Path, default=None,
                        help="Override output JSON path")
    args = parser.parse_args()

    OUT_PATH, OUT_LOG = _output_paths(args)

    # ── Load questions ────────────────────────────────────────────────────────
    def _load_and_normalise(path: Path, source_tag: str) -> list[dict]:
        """
        Load one of the two test-set JSONs and normalise to the internal shape
        expected by the rest of the harness:
          - question, answer, source, type, reviewed
          - chunk_id   → always a list (may be empty)
          - paper_ids  → always a list
        """
        with open(path) as f:
            items = json.load(f)
        out = []
        for item in items:
            # chunk_id is a string on single-paper, a list on cross-paper
            cid = item.get("chunk_id")
            if isinstance(cid, str):
                cid = [cid] if cid else []
            elif not isinstance(cid, list):
                cid = []

            # paper_ids: cross-paper has plural list; single-paper has singular string
            pids = item.get("paper_ids") or (
                [item["paper_id"]] if item.get("paper_id") else []
            )

            out.append({
                "question":     item["question"],
                "answer":       item.get("answer", ""),
                "source":       source_tag,
                "type":         item.get("type", "unknown"),
                "reviewed":     item.get("reviewed", True),
                "chunk_id":     cid,
                "paper_ids":    pids,
            })
        return out

    single_qs = _load_and_normalise(SINGLE_PATH, "single_paper")
    cross_qs  = _load_and_normalise(CROSS_PATH,  "cross_paper")

    # Filter unreviewed cross-paper items unless caller opts in
    if not args.include_unreviewed:
        cross_qs = [q for q in cross_qs if q["reviewed"]]

    if args.subset == "single":
        questions = single_qs
    elif args.subset == "cross":
        questions = cross_qs
    else:
        questions = single_qs + cross_qs

    if args.n:
        questions = questions[:args.n]

    synthesis_model = os.getenv("PAPERMIND_SYNTHESIS_MODEL", "claude-sonnet-4-5-20250929")

    if args.agent:
        # Shorten model name for display: "claude-sonnet-4-5-..." → "sonnet-4-5"
        _parts = synthesis_model.replace("claude-", "").split("-20")[0]
        mode_parts = [f"LangGraph-agent({_parts})"]
    else:
        mode_parts = ["hybrid(dense+BM25+RRF)"]
        if args.rerank:
            mode_parts.append("+Cohere-rerank")
        if args.graph:
            mode_parts.append("+graph")
    mode_label = " ".join(mode_parts)

    console.print(f"\n[bold cyan]PaperMind — RAGAS Eval[/bold cyan]")
    console.print(
        f"Mode: {mode_label} | Questions: {len(questions)} | "
        f"subset={args.subset} | n={args.n or 'all'} | "
        f"synthesis={synthesis_model}"
    )
    reviewed_note = "reviewed only" if not args.include_unreviewed else "incl. unreviewed"
    console.print(
        f"Golden set: {len(single_qs)} single-paper | "
        f"{len(cross_qs)} cross-paper ({reviewed_note})\n"
    )
    console.print(f"[dim]Output → {OUT_PATH}[/dim]\n")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    checkpoint = {} if args.fresh else _load_checkpoint(OUT_PATH)
    existing_records = checkpoint.get("records", [])
    if existing_records and not args.fresh:
        console.print(
            f"[dim]Checkpoint found: {len(existing_records)} records in {OUT_PATH}[/dim]"
        )

    # ── Pipeline run ──────────────────────────────────────────────────────────
    if args.skip_pipeline and existing_records:
        console.print("[yellow]--skip-pipeline: reusing existing pipeline outputs[/yellow]")
        records = existing_records[:len(questions)]
    elif args.agent:
        console.print("[bold]Step 1:[/bold] Running LangGraph agent on all questions...")
        records = run_agent_for_all(questions, existing_records, OUT_PATH)
        console.print(f"  [dim]Agent outputs saved → {OUT_PATH}[/dim]\n")
    else:
        console.print("[bold]Step 1:[/bold] Running RAG pipeline on all questions...")
        # Cohere trial key: 10 req/min → need ≥6 s between rerank calls.
        # Set 7 s as the floor; the backoff in _rerank handles burst bursts.
        q_sleep = 7.0 if args.rerank else 0.2

        with RAGPipeline(
            synthesis_model=synthesis_model,
            use_rerank=args.rerank,
            use_graph=args.graph,
        ) as pipeline:
            records = run_pipeline_for_all(
                questions, pipeline, existing_records, OUT_PATH,
                inter_question_sleep=q_sleep,
            )
        console.print(f"  [dim]Pipeline outputs saved → {OUT_PATH}[/dim]\n")

    # ── RAGAS setup ───────────────────────────────────────────────────────────
    console.print("[bold]Step 2:[/bold] Scoring with RAGAS (judge=gpt-4o-mini)...")
    oa_async  = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    judge_llm = llm_factory("gpt-4o-mini", provider="openai", client=oa_async, max_tokens=16000)
    ragas_emb = OpenAIEmbeddings(model="text-embedding-3-small", client=oa_async)

    f_metric  = Faithfulness(llm=judge_llm)
    ar_metric = AnswerRelevancy(llm=judge_llm, embeddings=ragas_emb)
    cr_metric = ContextRecall(llm=judge_llm)
    cp_metric = ContextPrecision(llm=judge_llm)

    scored = asyncio.run(
        score_all(records, f_metric, ar_metric, cr_metric, cp_metric, OUT_PATH)
    )
    agg  = aggregate(scored)
    lats = latency_stats(scored)

    # ── Results table ─────────────────────────────────────────────────────────
    targets = {
        "faithfulness":      0.85,
        "answer_relevancy":  0.80,
        "context_recall":    0.80,
        "context_precision": 0.70,
    }

    console.print(f"\n{'='*58}")
    console.print(f"[bold]RAGAS RESULTS — {mode_label}[/bold]")
    console.print(f"{'='*58}")

    tbl = Table(show_header=True, header_style="bold cyan")
    tbl.add_column("Metric",          width=22)
    tbl.add_column("Score",  justify="right", width=8)
    tbl.add_column("Target", justify="right", width=8)
    tbl.add_column("Gap",    justify="right", width=8)

    for metric, score in agg.items():
        if score is None:
            tbl.add_row(metric.replace("_", " ").title(), "—", "—", "—")
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
        f"pipeline={synthesis_model} | judge=gpt-4o-mini\n"
    )

    # ── Latency table ─────────────────────────────────────────────────────────
    console.print("[bold]Latency (ms)[/bold]")
    lat_tbl = Table(show_header=True, header_style="bold")
    lat_tbl.add_column("Stage",       width=16)
    lat_tbl.add_column("p50", justify="right", width=8)
    lat_tbl.add_column("p95", justify="right", width=8)

    lat_tbl.add_row(
        "Total",
        str(lats.get("total_p50_ms", "—")),
        str(lats.get("total_p95_ms", "—")),
    )
    for key in ("classifier_ms", "planner_ms", "retrieve_ms",
                "rerank_ms", "gate_ms", "replan_ms", "synthesis_ms",
                # naive pipeline stages (kept for --agent=False runs)
                "embed_ms", "dense_ms", "bm25_ms", "graph_ms", "fetch_ms"):
        p50 = lats.get(f"{key}_p50")
        p95 = lats.get(f"{key}_p95")
        if p50 is not None:
            lat_tbl.add_row(key.replace("_ms", ""), str(p50), str(p95))

    console.print(lat_tbl)

    # ── Per-question breakdown ────────────────────────────────────────────────
    detail = Table(show_header=True, header_style="bold", show_lines=True)
    detail.add_column("#",     width=3,  justify="right")
    detail.add_column("Src",   width=4)
    detail.add_column("Type",  width=11)
    detail.add_column("Question",           max_width=40)
    detail.add_column("Conf",  width=4,  justify="center")
    detail.add_column("Faith", width=6,  justify="right")
    detail.add_column("Rel",   width=5,  justify="right")
    detail.add_column("Rec",   width=5,  justify="right")
    detail.add_column("Prec",  width=5,  justify="right")
    detail.add_column("ms",    width=6,  justify="right")

    for i, rec in enumerate(scored, 1):
        faith  = rec.get("faithfulness")
        rel    = rec.get("answer_relevancy")
        recall = rec.get("context_recall")
        prec   = rec.get("context_precision")
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
            f"{faith:.2f}"  if faith  is not None else "—",
            f"{rel:.2f}"    if rel    is not None else "—",
            f"{recall:.2f}" if recall is not None else "—",
            f"{prec:.2f}"   if prec   is not None else "—",
            f"{rec['latency_ms']:.0f}",
        )

    console.print(detail)

    # ── Save full results ─────────────────────────────────────────────────────
    _save_checkpoint(OUT_PATH, scored, {
        "aggregates":     agg,
        "latency_stats":  lats,
        "n":              len(scored),
        "subset":         args.subset,
        "mode":           mode_label,
        "use_rerank":     args.rerank,
        "use_graph":      args.graph,
        "pipeline_model": synthesis_model,
        "judge_model":    "gpt-4o-mini",
        "ragas_version":  "0.4.3",
    })

    # ── Save terminal log ─────────────────────────────────────────────────────
    OUT_LOG.parent.mkdir(exist_ok=True)
    with open(OUT_LOG, "w") as f:
        f.write(console.export_text(clear=False))
    console.print(f"\n[dim]Full results → {OUT_PATH}[/dim]")
    console.print(f"[dim]Terminal log  → {OUT_LOG}[/dim]")
    langfuse_shutdown()   # flush any pending spans before process exits

    # ── One-liner for DESIGN.md ───────────────────────────────────────────────
    console.print("\n[bold]DESIGN.md entry:[/bold]")
    parts = [f"{k.replace('_', ' ').title()}: {v:.3f}" for k, v in agg.items() if v]
    console.print(f"  {mode_label} | {' | '.join(parts)}")
    rerank_note = (
        f" | rerank p50={lats.get('rerank_ms_p50')}ms p95={lats.get('rerank_ms_p95')}ms"
        if args.rerank else ""
    )
    console.print(
        f"  Latency — total p50={lats.get('total_p50_ms')}ms "
        f"p95={lats.get('total_p95_ms')}ms{rerank_note}"
    )


if __name__ == "__main__":
    main()
