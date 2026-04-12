"""
Phase 0 connection verification script.
Run after `docker compose up -d` to confirm all three services are reachable.

Usage:
    python scripts/verify_services.py
"""

import os
import sys
import httpx
from dotenv import load_dotenv
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from rich.console import Console
from rich.table import Table

load_dotenv()

console = Console()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
LANGFUSE_URL = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

TEST_COLLECTION = "papermind_verify_test"


def check_qdrant() -> tuple[bool, str]:
    try:
        client = QdrantClient(url=QDRANT_URL, timeout=5)
        # Create a temp collection and delete it
        client.create_collection(
            collection_name=TEST_COLLECTION,
            vectors_config=VectorParams(size=4, distance=Distance.COSINE),
        )
        client.delete_collection(TEST_COLLECTION)
        return True, "accessible, collection create/delete OK"
    except Exception as e:
        return False, str(e)


def check_neo4j() -> tuple[bool, str]:
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run("RETURN 1 AS n")
            result.single()
        driver.close()
        return True, "accessible, Cypher query OK"
    except Exception as e:
        return False, str(e)


def check_langfuse() -> tuple[bool, str]:
    try:
        r = httpx.get(f"{LANGFUSE_URL}/api/public/health", timeout=5)
        if r.status_code == 200:
            return True, f"accessible, health endpoint returned {r.status_code}"
        return False, f"unexpected status {r.status_code}"
    except Exception as e:
        return False, str(e)


def main() -> None:
    console.print("\n[bold]PaperMind — Phase 0 service verification[/bold]\n")

    checks = [
        ("Qdrant", f"{QDRANT_URL}", check_qdrant),
        ("Neo4j", f"{NEO4J_URI}", check_neo4j),
        ("Langfuse", f"{LANGFUSE_URL}", check_langfuse),
    ]

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Service", style="bold")
    table.add_column("Address")
    table.add_column("Status")
    table.add_column("Detail")

    all_ok = True
    for name, addr, fn in checks:
        ok, detail = fn()
        status = "[green]OK[/green]" if ok else "[red]FAIL[/red]"
        table.add_row(name, addr, status, detail)
        if not ok:
            all_ok = False

    console.print(table)

    if all_ok:
        console.print("\n[green bold]All services verified. Phase 0 complete.[/green bold]\n")
    else:
        console.print("\n[red bold]Some services failed. Check docker compose logs.[/red bold]\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
