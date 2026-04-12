"""
Post-processing fixes for Neo4j graph quality issues.

Fixes:
  1. Entity deduplication — merge nodes with same normalized name+type,
     and collapse semantic synonyms (RAG variants, LLM variants, etc.)
  2. Memory paper authors — remove institution names that leaked in
  3. (CITES_PAPER is handled by re-running citation_graph.py after code fix)

Run:
    python scripts/fix_neo4j.py
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from rich.console import Console

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

console = Console()

# ── Synonym map: canonical_name → [aliases that should collapse into it] ───────
# Key = the entity name we KEEP (already lowercase)
# Values = other lowercase names that mean the same thing

SYNONYMS: dict[str, list[str]] = {
    "rag": [
        "retrieval-augmented generation",
        "retrieval augmented generation",
        "retrieval-augmented generation (rag)",
        "retrieval augmented generation (rag)",
    ],
    "llm": [
        "llms",
        "large language model",
        "large language models",
    ],
    "agentic rag": [
        "agentic retrieval-augmented generation",
        "agentic retrieval augmented generation",
    ],
    "reinforcement learning": [
        "rl",
    ],
    "knowledge graph": [
        "knowledge graphs",
        "kg",
    ],
    "bm25": [
        "okapi bm25",
        "rank_bm25",
    ],
    "question answering": [
        "open-domain question answering",
        "open domain question answering",
    ],
    "retrieval": [
        "information retrieval",
    ],
    "modular rag": [
        "modular-rag",
    ],
    "graph rag": [
        "graph-rag",
        "graphrag",
        "graph retrieval-augmented generation",
    ],
    "corrective rag": [
        "crag",
    ],
    "adaptive rag": [
        "adaptive-rag",
        "adaptive retrieval-augmented generation",
    ],
    "colbert": [
        "colbertv2",
    ],
    "multi-hop reasoning": [
        "multi-hop retrieval",
        "multi-hop queries",
    ],
    "dense retrieval": [
        "dense passage retrieval",
        "dpr",
        "dense vector search",
        "dense semantic matching",
    ],
}

# ── Known institution names to strip from Memory paper authors ────────────────
KNOWN_INSTITUTIONS = {
    "CUHK-Shenzhen", "CUHK", "HITSZ", "BIT", "Huawei Cloud",
}


def merge_entities_by_synonym(session) -> int:
    """Merge synonym entities — redirect MENTIONS edges then delete duplicates."""
    merged = 0
    for canonical, aliases in SYNONYMS.items():
        for alias in aliases:
            # For each entity type, merge alias into canonical
            for etype in ["Method", "Task", "Dataset", "Metric"]:
                # Check if both exist
                result = session.run(
                    """
                    MATCH (keep:Entity {name: $canonical, type: $type})
                    MATCH (del:Entity {name: $alias, type: $type})
                    WHERE keep <> del
                    RETURN elementId(keep) AS keep_id, elementId(del) AS del_id,
                           keep.display_name AS keep_display
                    """,
                    canonical=canonical, alias=alias, type=etype,
                ).data()

                for row in result:
                    # Redirect all MENTIONS from del to keep
                    session.run(
                        """
                        MATCH (p:Paper)-[old:MENTIONS]->(del:Entity)
                        WHERE elementId(del) = $del_id
                        MATCH (keep:Entity)
                        WHERE elementId(keep) = $keep_id
                        MERGE (p)-[:MENTIONS {chunk_id: old.chunk_id, section: old.section}]->(keep)
                        DELETE old
                        """,
                        del_id=row["del_id"], keep_id=row["keep_id"],
                    )
                    # Delete orphaned entity
                    session.run(
                        """
                        MATCH (del:Entity)
                        WHERE elementId(del) = $del_id
                        DETACH DELETE del
                        """,
                        del_id=row["del_id"],
                    )
                    console.print(f"  Merged [{etype}] '{alias}' → '{canonical}'")
                    merged += 1
    return merged


def merge_same_name_diff_type(session) -> int:
    """
    Merge entities with the same canonical name but different types.
    Keeps type='Method' over 'Task', etc.
    Priority: Method > Dataset > Metric > Task
    """
    TYPE_PRIORITY = {"Method": 0, "Dataset": 1, "Metric": 2, "Task": 3}
    merged = 0

    rows = session.run(
        """
        MATCH (e:Entity)
        WITH e.name AS name, collect(e) AS nodes, count(e) AS cnt
        WHERE cnt > 1
        RETURN name, [n IN nodes | {id: elementId(n), type: n.type, display: n.display_name}] AS nodes
        """
    ).data()

    for row in rows:
        nodes = sorted(row["nodes"], key=lambda n: TYPE_PRIORITY.get(n["type"], 99))
        keep = nodes[0]
        for del_node in nodes[1:]:
            session.run(
                """
                MATCH (p:Paper)-[old:MENTIONS]->(del:Entity)
                WHERE elementId(del) = $del_id
                MATCH (keep:Entity)
                WHERE elementId(keep) = $keep_id
                MERGE (p)-[:MENTIONS {chunk_id: old.chunk_id, section: old.section}]->(keep)
                DELETE old
                """,
                del_id=del_node["id"], keep_id=keep["id"],
            )
            session.run(
                "MATCH (del:Entity) WHERE elementId(del) = $del_id DETACH DELETE del",
                del_id=del_node["id"],
            )
            console.print(
                f"  Merged same-name diff-type: '{row['name']}' "
                f"[{del_node['type']}] → [{keep['type']}]"
            )
            merged += 1

    return merged


def fix_memory_paper_authors(session) -> None:
    """Remove institution names that leaked into the Memory paper's author list."""
    result = session.run(
        "MATCH (p:Paper {paper_id: 'f508af927f8b'}) RETURN p.authors AS authors"
    ).single()

    if not result:
        console.print("  [yellow]Memory paper not found[/yellow]")
        return

    old_authors = result["authors"]
    clean_authors = [a for a in old_authors if a not in KNOWN_INSTITUTIONS]

    session.run(
        "MATCH (p:Paper {paper_id: 'f508af927f8b'}) SET p.authors = $authors",
        authors=clean_authors,
    )
    console.print(f"  Before: {old_authors}")
    console.print(f"  After : {clean_authors}")


def main() -> None:
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )

    with driver.session() as session:
        # ── 1. Entity synonym merging ─────────────────────────────────────────
        console.print("\n[bold]Fix 1:[/bold] Merging synonym entities...")
        n1 = merge_entities_by_synonym(session)
        console.print(f"  → {n1} synonym merges done")

        # ── 2. Same name, different type merging ──────────────────────────────
        console.print("\n[bold]Fix 2:[/bold] Merging same-name different-type entities...")
        n2 = merge_same_name_diff_type(session)
        console.print(f"  → {n2} type-conflict merges done")

        # ── 3. Fix Memory paper authors ───────────────────────────────────────
        console.print("\n[bold]Fix 3:[/bold] Cleaning Memory paper authors...")
        fix_memory_paper_authors(session)

        # ── Final counts ──────────────────────────────────────────────────────
        console.print("\n[bold]Final graph state:[/bold]")
        for label in ["Paper", "Entity", "Reference"]:
            n = session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
            console.print(f"  {label:12s}: {n}")
        for rel in ["MENTIONS", "CITES"]:
            n = session.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c").single()["c"]
            console.print(f"  {rel:12s}: {n}")

        # How many entities are now shared across 2+ papers?
        shared = session.run("""
            MATCH (p:Paper)-[:MENTIONS]->(e:Entity)
            WITH e, count(DISTINCT p) AS cnt
            WHERE cnt >= 2
            RETURN count(e) AS shared_entities
        """).single()["shared_entities"]
        console.print(f"\n  Entities shared by 2+ papers: [bold green]{shared}[/bold green]")

    driver.close()


if __name__ == "__main__":
    main()
