"""
Paper management endpoints for PaperMind.

GET  /papers              — list all indexed papers
POST /papers/upload       — upload a PDF and ingest it (steps 1-5, no Neo4j)
DELETE /papers/{paper_id} — remove a paper from Qdrant + rebuild BM25
"""

from __future__ import annotations

import asyncio
import hashlib
import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

router = APIRouter(prefix="/papers", tags=["papers"])

COLLECTION_NAME = "papers"
PAPERS_DIR = Path("data/papers")
MAX_PAPERS = 10


def _get_qdrant() -> QdrantClient:
    import os
    return QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
    )


def _paper_id_from_filename(filename: str) -> str:
    return hashlib.md5(filename.encode()).hexdigest()[:12]


def _scroll_all_points(client: QdrantClient) -> list[Any]:
    """Scroll through all points in the collection, returning all records."""
    all_points = []
    offset = None
    while True:
        result, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=None,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(result)
        if next_offset is None:
            break
        offset = next_offset
    return all_points


def _build_paper_list(points: list[Any]) -> list[dict]:
    """Aggregate Qdrant points into one record per paper."""
    by_paper: dict[str, dict] = {}
    for pt in points:
        p = pt.payload or {}
        pid = p.get("paper_id", "")
        if not pid:
            continue
        if pid not in by_paper:
            by_paper[pid] = {
                "id": pid,
                "title": p.get("paper_title", "Unknown"),
                "authors": p.get("authors", []),
                "year": p.get("year"),
                "sections": set(),
                "chunkCount": 0,
            }
        entry = by_paper[pid]
        entry["chunkCount"] += 1
        if p.get("section"):
            entry["sections"].add(p["section"])

    result = []
    for entry in by_paper.values():
        result.append({
            **entry,
            "sections": sorted(entry["sections"]),
        })
    return sorted(result, key=lambda p: p["title"])


# ── Response models ───────────────────────────────────────────────────────────

class PaperOut(BaseModel):
    id: str
    title: str
    authors: list[str]
    year: int | None
    sections: list[str]
    chunkCount: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("", response_model=list[PaperOut])
def list_papers():
    """Return all indexed papers (aggregated from Qdrant chunk payloads)."""
    try:
        client = _get_qdrant()
        points = _scroll_all_points(client)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Qdrant unavailable: {exc}")

    return _build_paper_list(points)


@router.post("/upload", response_model=PaperOut, status_code=201)
async def upload_paper(file: UploadFile):
    """
    Upload a PDF, save it to data/papers/, and run ingestion steps 1-5
    (parse → chunk → metadata → embed → BM25 rebuild).  Neo4j skipped.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # ── Check capacity ────────────────────────────────────────────────────────
    try:
        client = _get_qdrant()
        points = _scroll_all_points(client)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Qdrant unavailable: {exc}")

    existing_papers = _build_paper_list(points)
    if len(existing_papers) >= MAX_PAPERS:
        raise HTTPException(
            status_code=409,
            detail=f"Maximum of {MAX_PAPERS} papers reached. Remove a paper first.",
        )

    # ── Save PDF ──────────────────────────────────────────────────────────────
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    dest = PAPERS_DIR / file.filename
    content = await file.read()
    dest.write_bytes(content)

    # ── Run ingestion in a thread (blocking CPU work) ─────────────────────────
    try:
        paper_data = await asyncio.get_event_loop().run_in_executor(
            None, _ingest_single_paper, dest
        )
    except Exception as exc:
        # Clean up saved file on failure
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

    return PaperOut(**paper_data)


@router.delete("/{paper_id}", status_code=204)
def delete_paper(paper_id: str):
    """
    Delete all Qdrant points for paper_id and rebuild the BM25 index
    from the remaining chunks.
    """
    try:
        client = _get_qdrant()

        # Verify paper exists
        existing = _scroll_all_points(client)
        paper_ids = {pt.payload.get("paper_id") for pt in existing if pt.payload}
        if paper_id not in paper_ids:
            raise HTTPException(status_code=404, detail="Paper not found.")

        # Delete from Qdrant
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="paper_id",
                            match=qmodels.MatchValue(value=paper_id),
                        )
                    ]
                )
            ),
        )

        # Remove PDF from disk (best-effort — paper_id is md5 of filename)
        for pdf in PAPERS_DIR.glob("*.pdf"):
            if _paper_id_from_filename(pdf.name) == paper_id:
                pdf.unlink(missing_ok=True)
                break

        # Rebuild BM25 from remaining Qdrant chunks
        remaining = _scroll_all_points(client)
        if remaining:
            _rebuild_bm25_from_points(remaining)
        else:
            # Clear index if no papers remain
            import json, os
            bm25_dir = Path("data/bm25")
            mapping = bm25_dir / "chunk_ids.json"
            if mapping.exists():
                mapping.write_text(json.dumps([]))

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Ingestion helpers (run in executor) ───────────────────────────────────────

def _ingest_single_paper(pdf_path: Path) -> dict:
    """
    Run ingestion steps 1-5 for a single PDF.
    Returns paper metadata dict suitable for PaperOut.
    """
    from ingestion.bm25_index import BM25Index
    from ingestion.chunker import PaperChunker
    from ingestion.embedder import ChunkEmbedder
    from ingestion.metadata import extract_bibliography, extract_paper_metadata
    from ingestion.parser import PaperParser

    parser = PaperParser()
    paper = parser.parse(pdf_path)

    meta = extract_paper_metadata(paper)
    chunker = PaperChunker(chunk_size=512, chunk_overlap=64)
    chunks = chunker.chunk(
        paper,
        title=meta["title"],
        authors=meta["authors"],
        year=meta["year"],
    )

    embedder = ChunkEmbedder()
    embedder.embed_and_store(chunks)

    # Rebuild BM25 from ALL points (including the new paper)
    all_points = _scroll_all_points(embedder._qdrant)
    _rebuild_bm25_from_points(all_points)

    sections = sorted({c.section for c in chunks if c.section})
    return {
        "id": paper.paper_id,
        "title": meta["title"] or pdf_path.stem,
        "authors": meta["authors"],
        "year": meta["year"],
        "sections": sections,
        "chunkCount": len(chunks),
    }


def _rebuild_bm25_from_points(points: list[Any]) -> None:
    """Reconstruct Chunk objects from Qdrant payloads and rebuild BM25 index."""
    from ingestion.bm25_index import BM25Index
    from ingestion.models import Chunk

    chunks = []
    for pt in points:
        p = pt.payload or {}
        if not p.get("chunk_id"):
            continue
        chunks.append(
            Chunk(
                chunk_id=p["chunk_id"],
                paper_id=p.get("paper_id", ""),
                paper_title=p.get("paper_title", ""),
                authors=p.get("authors", []),
                year=p.get("year"),
                section=p.get("section", ""),
                chunk_index=p.get("chunk_index", 0),
                text=p.get("text", ""),
                is_table=p.get("is_table", False),
                contains_math=p.get("contains_math", False),
            )
        )

    bm25 = BM25Index()
    bm25.build(chunks, show_progress=False)
