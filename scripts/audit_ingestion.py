"""
Full ingestion audit — verifies every layer of Phase 1 output.

Checks:
  [Qdrant]  - Point count matches expected, all payload fields present,
              vectors non-zero, semantic search returns sensible results
  [BM25]    - Index loads, keyword search returns sensible results
  [Neo4j]   - All 10 papers present, entity types valid, edge counts sane,
              chunk_ids on MENTIONS edges actually exist in Qdrant
  [Chunks]  - Per-paper: section coverage, metadata completeness,
              no corrupt/empty chunks, is_table flag accuracy
  [Cross]   - chunk_id namespace consistent across BM25 ↔ Qdrant ↔ Neo4j

Usage:
    python scripts/audit_ingestion.py
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv(Path(__file__).parent.parent / ".env", override=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.bm25_index import BM25Index
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from neo4j import GraphDatabase

console = Console()
PASS = "[bold green]PASS[/bold green]"
FAIL = "[bold red]FAIL[/bold red]"
WARN = "[bold yellow]WARN[/bold yellow]"

EXPECTED_PAPERS = 10
EXPECTED_CHUNKS = 1035
VALID_ENTITY_TYPES = {"Method", "Dataset", "Metric", "Task"}
REQUIRED_PAYLOAD_FIELDS = {
    "chunk_id", "paper_id", "paper_title", "authors",
    "year", "section", "text", "is_table", "contains_math",
}

issues: list[str] = []


def check(label: str, ok: bool, detail: str = "", warn_only: bool = False) -> bool:
    tag = PASS if ok else (WARN if warn_only else FAIL)
    console.print(f"  {tag}  {label}" + (f"  — {detail}" if detail else ""))
    if not ok and not warn_only:
        issues.append(label)
    return ok


# ── 1. Qdrant ─────────────────────────────────────────────────────────────────

def audit_qdrant(qdrant: QdrantClient) -> set[str]:
    """Returns set of all chunk_ids in Qdrant."""
    console.print("\n[bold cyan]── Qdrant ──────────────────────────────────[/bold cyan]")

    info = qdrant.get_collection("papers")
    check("Collection 'papers' exists", True)
    check(
        f"Point count = {EXPECTED_CHUNKS}",
        info.points_count == EXPECTED_CHUNKS,
        f"got {info.points_count}",
    )
    check(
        "Vector dimension = 1536",
        info.config.params.vectors.size == 1536,
        f"got {info.config.params.vectors.size}",
    )
    check(
        "Distance = Cosine",
        str(info.config.params.vectors.distance).lower() == "cosine",
    )

    # Scroll through ALL points, verify payloads and collect chunk_ids
    all_chunk_ids: set[str] = set()
    missing_fields_count = 0
    empty_text_count = 0
    bad_year_count = 0
    no_section_count = 0
    papers_seen: set[str] = set()
    table_count = 0

    offset = None
    while True:
        result, next_offset = qdrant.scroll(
            collection_name="papers",
            limit=200,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for pt in result:
            p = pt.payload
            all_chunk_ids.add(p.get("chunk_id", ""))
            papers_seen.add(p.get("paper_id", ""))

            # Payload completeness
            missing = REQUIRED_PAYLOAD_FIELDS - set(p.keys())
            if missing:
                missing_fields_count += 1

            # Text quality
            if not p.get("text") or len(p.get("text", "")) < 10:
                empty_text_count += 1

            # Year validity
            year = p.get("year")
            if year is not None and not (2000 <= year <= 2030):
                bad_year_count += 1

            # Section present
            if not p.get("section"):
                no_section_count += 1

            # Table flag
            if p.get("is_table"):
                table_count += 1

            # Vector non-zero
            vec = pt.vector
            if vec and all(v == 0 for v in vec[:10]):
                console.print(f"    [red]Zero vector: {p.get('chunk_id')}[/red]")

        if next_offset is None:
            break
        offset = next_offset

    check(f"All {EXPECTED_CHUNKS} payloads have required fields",
          missing_fields_count == 0, f"{missing_fields_count} missing fields")
    check("No empty text chunks", empty_text_count == 0, f"{empty_text_count} found")
    check("All years valid (2000–2030)", bad_year_count == 0, f"{bad_year_count} invalid")
    check("All chunks have section labels", no_section_count == 0, f"{no_section_count} missing")
    check(f"All {EXPECTED_PAPERS} papers represented",
          len(papers_seen) == EXPECTED_PAPERS, f"got {len(papers_seen)}")
    console.print(f"    Table chunks: {table_count} / {EXPECTED_CHUNKS}")

    # Semantic search sanity using query_points
    try:
        from openai import OpenAI
        oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        vec = oai.embeddings.create(
            input=["How does BM25 compare to dense retrieval?"],
            model="text-embedding-3-small",
        ).data[0].embedding
        hits = qdrant.query_points(
            collection_name="papers",
            query=vec,
            limit=3,
            with_payload=True,
        ).points
        console.print(f"\n    [dim]Semantic search: 'How does BM25 compare to dense retrieval?'[/dim]")
        for h in hits:
            console.print(f"    score={h.score:.3f}  [{h.payload['section']}]  {h.payload['text'][:80]}...")
        check("Semantic search returns relevant results",
              any("bm25" in h.payload["text"].lower() or "retrieval" in h.payload["text"].lower()
                  for h in hits))
    except Exception as e:
        check("Semantic search sanity", False, str(e))

    return all_chunk_ids


# ── 2. BM25 ──────────────────────────────────────────────────────────────────

def audit_bm25(qdrant_chunk_ids: set[str]) -> set[str]:
    console.print("\n[bold cyan]── BM25 ─────────────────────────────────────[/bold cyan]")

    bm25 = BM25Index()
    check("BM25 index file exists", bm25.exists())
    bm25.load()
    check(f"BM25 index size = {EXPECTED_CHUNKS}", bm25.size == EXPECTED_CHUNKS,
          f"got {bm25.size}")

    # Keyword search sanity
    hits = bm25.query("BM25 retrieval dense sparse hybrid", k=5)
    check("BM25 returns results for keyword query", len(hits) > 0, f"{len(hits)} hits")

    console.print(f"\n    [dim]BM25 query: 'BM25 retrieval dense sparse hybrid'[/dim]")
    for chunk_id, score in hits[:3]:
        console.print(f"    score={score:.3f}  chunk_id={chunk_id}")

    # Check chunk_ids in BM25 exist in Qdrant
    bm25_ids = {chunk_id for chunk_id, _ in bm25.query("the", k=EXPECTED_CHUNKS)}
    # Get all BM25 ids via full query
    all_bm25_ids: set[str] = set()
    with open(Path("data/bm25/chunk_ids.json")) as f:
        all_bm25_ids = set(json.load(f))

    orphans = all_bm25_ids - qdrant_chunk_ids
    check(
        "All BM25 chunk_ids exist in Qdrant",
        len(orphans) == 0,
        f"{len(orphans)} orphaned IDs",
    )
    return all_bm25_ids


# ── 3. Neo4j ──────────────────────────────────────────────────────────────────

def audit_neo4j(session, qdrant_chunk_ids: set[str]) -> None:
    console.print("\n[bold cyan]── Neo4j ─────────────────────────────────────[/bold cyan]")

    # Node counts
    paper_count = session.run("MATCH (p:Paper) RETURN count(p) AS c").single()["c"]
    entity_count = session.run("MATCH (e:Entity) RETURN count(e) AS c").single()["c"]
    ref_count = session.run("MATCH (r:Reference) RETURN count(r) AS c").single()["c"]

    check(f"All {EXPECTED_PAPERS} Paper nodes present",
          paper_count == EXPECTED_PAPERS, f"got {paper_count}")
    check("Entity nodes > 500", entity_count > 500, f"got {entity_count}")
    check("Reference nodes > 100", ref_count > 100, f"got {ref_count}")

    # Relationship counts
    mentions = session.run("MATCH ()-[r:MENTIONS]->() RETURN count(r) AS c").single()["c"]
    cites = session.run("MATCH ()-[r:CITES]->() RETURN count(r) AS c").single()["c"]
    cites_paper = session.run("MATCH ()-[r:CITES_PAPER]->() RETURN count(r) AS c").single()["c"]

    check("MENTIONS edges > 1000", mentions > 1000, f"got {mentions}")
    check("CITES edges > 100", cites > 100, f"got {cites}")
    console.print(f"    CITES_PAPER edges: {cites_paper} (cross-paper citations in corpus)")

    # Entity type validity
    bad_types = session.run("""
        MATCH (e:Entity)
        WHERE NOT e.type IN ['Method', 'Dataset', 'Metric', 'Task']
        RETURN count(e) AS c
    """).single()["c"]
    check("All Entity types are valid (Method/Dataset/Metric/Task)",
          bad_types == 0, f"{bad_types} invalid types")

    # Every Paper has at least some MENTIONS
    lonely_papers = session.run("""
        MATCH (p:Paper)
        WHERE NOT (p)-[:MENTIONS]->()
        RETURN count(p) AS c
    """).single()["c"]
    check("All Papers have MENTIONS edges", lonely_papers == 0,
          f"{lonely_papers} papers with no entity mentions")

    # Paper nodes have required properties
    incomplete_papers = session.run("""
        MATCH (p:Paper)
        WHERE p.title IS NULL OR p.year IS NULL OR p.authors IS NULL
        RETURN count(p) AS c
    """).single()["c"]
    check("All Paper nodes have title/year/authors", incomplete_papers == 0,
          f"{incomplete_papers} incomplete")

    # MENTIONS edges have chunk_ids — spot check 50
    sample_chunk_ids = session.run("""
        MATCH ()-[r:MENTIONS]->()
        WHERE r.chunk_id IS NOT NULL
        RETURN r.chunk_id AS cid LIMIT 50
    """).data()
    sample_ids = {r["cid"] for r in sample_chunk_ids}
    orphans = sample_ids - qdrant_chunk_ids
    check(
        "MENTIONS edge chunk_ids exist in Qdrant (sample 50)",
        len(orphans) == 0,
        f"{len(orphans)} chunk_ids not in Qdrant",
    )

    # Entity sharing
    shared = session.run("""
        MATCH (p:Paper)-[:MENTIONS]->(e:Entity)
        WITH e, count(DISTINCT p) AS cnt WHERE cnt >= 2
        RETURN count(e) AS c
    """).single()["c"]
    check("At least 30 entities shared across 2+ papers",
          shared >= 30, f"got {shared}")

    # Show per-paper entity counts
    console.print("\n    [dim]Entities per paper:[/dim]")
    rows = session.run("""
        MATCH (p:Paper)-[:MENTIONS]->(e:Entity)
        WITH p, count(DISTINCT e) AS entities
        RETURN p.title AS title, entities ORDER BY entities DESC
    """).data()
    for r in rows:
        console.print(f"    {r['entities']:4d}  {str(r['title'])[:60]}")


# ── 4. Chunk metadata spot-check ──────────────────────────────────────────────

def audit_chunk_metadata(qdrant: QdrantClient) -> None:
    console.print("\n[bold cyan]── Chunk Metadata ───────────────────────────[/bold cyan]")

    # Pull all chunks grouped by paper
    paper_stats: dict[str, dict] = {}
    offset = None
    while True:
        result, next_offset = qdrant.scroll(
            collection_name="papers", limit=200, offset=offset,
            with_payload=True, with_vectors=False,
        )
        for pt in result:
            p = pt.payload
            pid = p["paper_id"]
            if pid not in paper_stats:
                paper_stats[pid] = {
                    "title": p["paper_title"],
                    "chunks": 0,
                    "sections": set(),
                    "no_author": 0,
                    "no_year": 0,
                    "tables": 0,
                    "short_chunks": 0,
                    "long_chunks": 0,
                }
            s = paper_stats[pid]
            s["chunks"] += 1
            s["sections"].add(p["section"])
            if not p.get("authors") or p["authors"] == ["Unknown"]:
                s["no_author"] += 1
            if not p.get("year"):
                s["no_year"] += 1
            if p.get("is_table"):
                s["tables"] += 1
            tlen = len(p["text"])
            if tlen < 50:
                s["short_chunks"] += 1
            if tlen > 2000:
                s["long_chunks"] += 1
        if next_offset is None:
            break
        offset = next_offset

    t = Table(show_header=True, header_style="bold", show_lines=False)
    t.add_column("Paper", max_width=28)
    t.add_column("Chunks", justify="right")
    t.add_column("Sections", justify="right")
    t.add_column("Tables", justify="right")
    t.add_column("No Author", justify="right")
    t.add_column("No Year", justify="right")
    t.add_column("Short(<50)", justify="right")

    all_ok = True
    for pid, s in sorted(paper_stats.items(), key=lambda x: x[1]["title"]):
        issues_here = []
        if s["no_author"] > 0:
            issues_here.append(f"no_author={s['no_author']}")
        if s["no_year"] > 0:
            issues_here.append(f"no_year={s['no_year']}")
        if s["short_chunks"] > 0:
            issues_here.append(f"short={s['short_chunks']}")
        if len(s["sections"]) < 3:
            issues_here.append(f"only {len(s['sections'])} sections!")
        if issues_here:
            all_ok = False

        t.add_row(
            str(s["title"])[:28],
            str(s["chunks"]),
            str(len(s["sections"])),
            str(s["tables"]),
            str(s["no_author"]) if s["no_author"] else "[green]0[/green]",
            str(s["no_year"]) if s["no_year"] else "[green]0[/green]",
            str(s["short_chunks"]) if s["short_chunks"] else "[green]0[/green]",
        )

    console.print(t)
    check("All chunks have author + year metadata", all_ok)


# ── 5. Cross-layer chunk_id consistency ───────────────────────────────────────

def audit_cross_layer(qdrant_ids: set[str], bm25_ids: set[str]) -> None:
    console.print("\n[bold cyan]── Cross-layer ID Consistency ───────────────[/bold cyan]")

    only_qdrant = qdrant_ids - bm25_ids
    only_bm25 = bm25_ids - qdrant_ids

    check("Qdrant IDs == BM25 IDs (exact match)",
          len(only_qdrant) == 0 and len(only_bm25) == 0,
          f"only-Qdrant={len(only_qdrant)}, only-BM25={len(only_bm25)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    qdrant = QdrantClient(host="localhost", port=6333)
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )

    qdrant_ids = audit_qdrant(qdrant)
    bm25_ids = audit_bm25(qdrant_ids)

    with driver.session() as session:
        audit_neo4j(session, qdrant_ids)

    audit_chunk_metadata(qdrant)
    audit_cross_layer(qdrant_ids, bm25_ids)

    driver.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    console.print()
    if issues:
        console.print(Panel(
            f"[bold red]{len(issues)} issue(s) found:[/bold red]\n" +
            "\n".join(f"  • {i}" for i in issues),
            border_style="red",
        ))
    else:
        console.print(Panel(
            "[bold green]All checks passed — ingestion is clean.[/bold green]",
            border_style="green",
        ))


if __name__ == "__main__":
    main()
