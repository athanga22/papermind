"""
Auto-populate chunk_ids and paper_ids for verifiable question types.

Types processed:   factoid, limitation, false_premise, contradiction, table_extraction
Types skipped:     multi_section, comparison, synthesis, aggregation, adversarial

Logic per type:
  factoid / limitation  → top-1 hit (filtered to paper_ids if set)
  false_premise         → top-1 hit (the CORRECT fact must exist in corpus)
  contradiction         → top-1 hit per paper (needs chunks from both papers)

Entries that already have chunk_id filled are skipped unless --force.

Usage:
    python scripts/auto_populate_chunks.py eval/data/sample.json
    python scripts/auto_populate_chunks.py eval/data/sample.json --out eval/data/verified.json
    python scripts/auto_populate_chunks.py eval/data/sample.json --dry-run
    python scripts/auto_populate_chunks.py eval/data/sample.json --force   # re-check already filled
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv(Path(__file__).parent.parent / ".env", override=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

from query.retriever import TrimodalRetriever

VERIFIABLE_TYPES = {"factoid", "limitation", "false_premise", "contradiction", "table_extraction"}
TOP_K = 5   # retrieve top-5, then pick best match

console = Console()


def best_hit(chunks, paper_ids: list[str] | None):
    """
    Return top-1 chunk, preferring chunks whose paper_id is in paper_ids.
    Falls back to overall top-1 if no paper_ids match.
    """
    if not chunks:
        return None
    if paper_ids:
        filtered = [c for c in chunks if c.paper_id in paper_ids]
        if filtered:
            return filtered[0]
    return chunks[0]


def populate(items: list[dict], retriever: TrimodalRetriever, force: bool, dry_run: bool):
    skipped_type  = 0
    skipped_done  = 0
    processed     = 0
    no_hit        = 0

    results = []

    for item in items:
        q_type = item.get("type", "")

        # ── Skip non-verifiable types ─────────────────────────────────────────
        if q_type not in VERIFIABLE_TYPES:
            skipped_type += 1
            results.append(item)
            continue

        # ── Skip already populated (unless --force) ───────────────────────────
        existing_cids = item.get("chunk_id") or []
        if existing_cids and not force:
            skipped_done += 1
            results.append(item)
            continue

        question  = item["question"]
        paper_ids = item.get("paper_ids") or []
        if isinstance(paper_ids, str):
            paper_ids = [paper_ids] if paper_ids else []

        # false_premise: query by answer (contains the true value), not the question
        # (the question embeds a false claim that misleads retrieval)
        query = item.get("answer", question) if q_type == "false_premise" else question

        console.print(f"\n[cyan]Q:[/cyan] {question[:80]}...")
        console.print(f"   type={q_type}  paper_ids={paper_ids or '(any)'}  query_by={'answer' if q_type == 'false_premise' else 'question'}")

        chunks = retriever.retrieve(query, top_k=TOP_K)

        updated = dict(item)

        if q_type == "contradiction":
            # Need one chunk per paper — paper_ids should list both
            if len(paper_ids) >= 2:
                cids = []
                found_papers = []
                for pid in paper_ids:
                    hit = best_hit([c for c in chunks if c.paper_id == pid], [pid])
                    if hit:
                        cids.append(hit.chunk_id)
                        found_papers.append(hit.paper_id)
                        console.print(f"   [green]✓[/green] {hit.chunk_id} ({hit.paper_title[:50]}) score={hit.score:.4f}")
                    else:
                        console.print(f"   [red]✗[/red] no chunk found for paper_id={pid}")
                        no_hit += 1
                if cids:
                    updated["chunk_id"]   = cids
                    updated["paper_ids"]  = found_papers
                    updated["auto_populated"] = True
                    processed += 1
                else:
                    console.print("   [red]✗ no hits at all[/red]")
            else:
                # No paper_ids set — take top-2 from different papers
                seen_papers: dict[str, str] = {}
                for c in chunks:
                    if c.paper_id not in seen_papers:
                        seen_papers[c.paper_id] = c.chunk_id
                    if len(seen_papers) == 2:
                        break
                if seen_papers:
                    updated["chunk_id"]      = list(seen_papers.values())
                    updated["paper_ids"]     = list(seen_papers.keys())
                    updated["auto_populated"] = True
                    processed += 1
                    for pid, cid in seen_papers.items():
                        console.print(f"   [yellow]~[/yellow] {cid} (paper={pid})")
                else:
                    console.print("   [red]✗ no hits[/red]")
                    no_hit += 1

        else:
            # factoid / limitation / false_premise → top-1
            hit = best_hit(chunks, paper_ids if paper_ids else None)
            if hit:
                updated["chunk_id"]      = [hit.chunk_id]
                updated["paper_ids"]     = [hit.paper_id]
                updated["auto_populated"] = True
                processed += 1
                match_flag = "[green]✓[/green]" if (not paper_ids or hit.paper_id in paper_ids) else "[yellow]~[/yellow] (paper mismatch)"
                console.print(f"   {match_flag} {hit.chunk_id} | {hit.paper_title[:50]} | score={hit.score:.4f}")
                console.print(f"   section: {hit.section}")
                console.print(f"   text: {hit.text[:150]}...")
            else:
                console.print("   [red]✗ no hits[/red]")
                no_hit += 1

        results.append(updated)

    return results, {
        "processed":    processed,
        "skipped_type": skipped_type,
        "skipped_done": skipped_done,
        "no_hit":       no_hit,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input",          type=Path, help="Input JSON file")
    parser.add_argument("--out",          type=Path, default=None,
                        help="Output path (default: overwrites input)")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Print results without writing")
    parser.add_argument("--force",        action="store_true",
                        help="Re-check entries that already have chunk_ids")
    parser.add_argument("--types",        nargs="+", default=None,
                        help="Only process these types (e.g. --types factoid limitation)")
    args = parser.parse_args()

    if not args.input.exists():
        console.print(f"[red]File not found: {args.input}[/red]")
        sys.exit(1)

    with open(args.input) as f:
        items = json.load(f)

    console.print(f"\n[bold cyan]PaperMind — Auto Chunk Populator[/bold cyan]")
    console.print(f"Input : {args.input}  ({len(items)} items)")
    console.print(f"Mode  : {'dry-run' if args.dry_run else 'write'}  force={args.force}")

    # Override verifiable types if --types passed
    global VERIFIABLE_TYPES
    if args.types:
        VERIFIABLE_TYPES = set(args.types)

    type_counts = {}
    for item in items:
        t = item.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    tbl = Table(show_header=True, header_style="bold")
    tbl.add_column("Type",  width=20)
    tbl.add_column("Count", justify="right", width=6)
    tbl.add_column("Will process?", width=14)
    for t, cnt in sorted(type_counts.items()):
        will = "[green]yes[/green]" if t in VERIFIABLE_TYPES else "[dim]skip[/dim]"
        tbl.add_row(t, str(cnt), will)
    console.print(tbl)

    with TrimodalRetriever() as retriever:
        results, stats = populate(items, retriever, force=args.force, dry_run=args.dry_run)

    console.print(f"\n[bold]Summary[/bold]")
    console.print(f"  Processed   : {stats['processed']}")
    console.print(f"  Skipped (type not verifiable): {stats['skipped_type']}")
    console.print(f"  Skipped (already filled)     : {stats['skipped_done']}")
    console.print(f"  No hit found : [red]{stats['no_hit']}[/red]")

    if not args.dry_run:
        out_path = args.out or args.input
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"\n[green]Saved → {out_path}[/green]")
        console.print(f"[dim]Entries marked auto_populated=True need spot-check.[/dim]")
    else:
        console.print("\n[yellow]Dry-run: nothing written.[/yellow]")


if __name__ == "__main__":
    main()
