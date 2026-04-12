"""
Step 6 — Entity extraction with Claude Haiku + Neo4j graph storage.

Entity types extracted per chunk:
  - Method / Technique (e.g., "RAG", "HNSW", "BM25")
  - Dataset (e.g., "HotpotQA", "MSMARCO")
  - Metric (e.g., "F1", "NDCG@10", "Recall")
  - Task (e.g., "question answering", "document retrieval")

Graph schema (Neo4j):
  Nodes:
    (:Paper {paper_id, title, authors, year})
    (:Entity {name, type})          ← canonical, lower-cased name
  Edges:
    (:Paper)-[:MENTIONS {chunk_id, section}]->(:Entity)

Strategy:
  - Extract entities from a representative sample of chunks per paper (up to 30)
    to keep Haiku API cost low during dev. Full-paper mode toggleable via flag.
  - JSON extraction via structured prompt — no tool use required for this simple schema
  - Deduplication: entities are merged by (lower(name), type) before writing
  - Idempotent: MERGE on both Paper and Entity nodes; MERGE on MENTIONS edge
  - Rate limiting: 1 API call per chunk, sequential (no parallelism needed for 10 papers)
"""

import json
import os
import re
import time

import anthropic
from neo4j import GraphDatabase

from ingestion.models import Chunk

# ── Constants ─────────────────────────────────────────────────────────────────

HAIKU_MODEL = "claude-haiku-4-5-20251001"
MAX_CHUNKS_PER_PAPER = 30        # sample limit to control cost in dev
ENTITY_TYPES = ["Method", "Dataset", "Metric", "Task"]

_SYSTEM_PROMPT = """\
You are an information extraction assistant for academic AI/ML papers.
Extract named entities from the provided text chunk.

Return ONLY a JSON array of objects. Each object must have:
  "name": the entity name as it appears (keep original casing)
  "type": one of Method, Dataset, Metric, Task

Rules:
- Method: algorithms, architectures, techniques, systems (e.g. RAG, HNSW, BM25, BERT)
- Dataset: named evaluation or training datasets (e.g. HotpotQA, SQuAD, MSMARCO)
- Metric: evaluation metrics (e.g. F1, Recall@10, NDCG, MRR, accuracy)
- Task: NLP/ML tasks (e.g. question answering, retrieval, summarization)
- Skip generic words, author names, institution names, and conjunctions
- Return [] if no entities found

Example output: [{"name": "BM25", "type": "Method"}, {"name": "F1", "type": "Metric"}]
"""

_JSON_RE = re.compile(r"\[.*?\]", re.DOTALL)


def _parse_entities(raw: str) -> list[dict]:
    """Extract and validate JSON array from Haiku response."""
    m = _JSON_RE.search(raw)
    if not m:
        return []
    try:
        items = json.loads(m.group())
        return [
            {"name": str(e["name"]).strip(), "type": str(e["type"]).strip()}
            for e in items
            if isinstance(e, dict)
            and e.get("name")
            and e.get("type") in ENTITY_TYPES
        ]
    except (json.JSONDecodeError, KeyError):
        return []


# ── Entity Extractor ──────────────────────────────────────────────────────────

class EntityExtractor:
    """
    Extracts entities from paper chunks and writes them to Neo4j.
    One instance per pipeline run.
    """

    def __init__(self) -> None:
        self._claude = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self._driver = GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
        )
        self._ensure_constraints()

    def _ensure_constraints(self) -> None:
        """Create uniqueness constraints if they don't exist."""
        with self._driver.session() as session:
            session.run(
                "CREATE CONSTRAINT paper_id IF NOT EXISTS "
                "FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT entity_name_type IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE"
            )

    def _extract_from_chunk(self, chunk: Chunk) -> list[dict]:
        """Call Haiku to extract entities from one chunk. Returns list of {name, type}."""
        response = self._claude.messages.create(
            model=HAIKU_MODEL,
            max_tokens=512,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": chunk.text}],
        )
        return _parse_entities(response.content[0].text)

    def _write_paper_node(self, session, chunk: Chunk) -> None:
        session.run(
            """
            MERGE (p:Paper {paper_id: $paper_id})
            SET p.title = $title,
                p.authors = $authors,
                p.year = $year
            """,
            paper_id=chunk.paper_id,
            title=chunk.paper_title,
            authors=chunk.authors,
            year=chunk.year,
        )

    def _write_entity_and_edge(
        self,
        session,
        chunk: Chunk,
        entity_name: str,
        entity_type: str,
    ) -> None:
        canonical = entity_name.lower()
        session.run(
            """
            MERGE (e:Entity {name: $canonical, type: $type})
            ON CREATE SET e.display_name = $display_name
            WITH e
            MATCH (p:Paper {paper_id: $paper_id})
            MERGE (p)-[:MENTIONS {chunk_id: $chunk_id, section: $section}]->(e)
            """,
            canonical=canonical,
            type=entity_type,
            display_name=entity_name,
            paper_id=chunk.paper_id,
            chunk_id=chunk.chunk_id,
            section=chunk.section,
        )

    def process_paper(
        self,
        chunks: list[Chunk],
        full_paper: bool = False,
        verbose: bool = False,
    ) -> int:
        """
        Extract entities from chunks of one paper and write to Neo4j.
        - full_paper=False: samples up to MAX_CHUNKS_PER_PAPER text chunks
        - Returns count of entity mentions written.
        """
        if not chunks:
            return 0

        # Filter to text-only chunks (skip tables — less entity-dense, save cost)
        text_chunks = [c for c in chunks if not c.is_table]

        # Sample if not doing full-paper mode
        if not full_paper and len(text_chunks) > MAX_CHUNKS_PER_PAPER:
            step = len(text_chunks) // MAX_CHUNKS_PER_PAPER
            text_chunks = text_chunks[::step][:MAX_CHUNKS_PER_PAPER]

        mentions = 0

        with self._driver.session() as session:
            # Ensure paper node exists
            self._write_paper_node(session, chunks[0])

            for chunk in text_chunks:
                entities = self._extract_from_chunk(chunk)
                if verbose:
                    print(f"  [{chunk.chunk_id}] {len(entities)} entities")

                for entity in entities:
                    self._write_entity_and_edge(
                        session,
                        chunk,
                        entity["name"],
                        entity["type"],
                    )
                    mentions += 1

                # Small delay to avoid rate limiting
                time.sleep(0.1)

        return mentions

    def close(self) -> None:
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
