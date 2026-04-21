# PaperMind

An agentic RAG system for question-answering over a curated corpus of academic papers. It routes queries by complexity, retrieves across dense and sparse modalities, and synthesizes cited answers using Claude.

**Retrieval baseline:** 97% Hit@5, MRR=0.737 on 787 chunks across 10 papers.

---

## What it does

PaperMind answers research questions against a fixed corpus of 10 papers. It handles factoid lookups, methodology explanations, cross-paper comparisons, synthesis across 3+ papers, and adversarial questions (correctly refusing when the topic isn't in the corpus).

It is not a generalist search engine. The 10-paper scope is intentional — it lets you validate every component of an agentic retrieval pipeline on a controlled, well-understood corpus before scaling.

---

## Architecture

```
User query
    │
    ▼
┌─────────────┐
│  Classifier  │  Haiku — routes to simple/moderate/comparison/synthesis
└──────┬──────┘  Sets max_sub_queries (2/4/4/5) and target_papers
       │
       ▼
┌─────────────┐
│   Planner   │  Haiku — decomposes query into surgical retrieval strings
└──────┬──────┘  Skipped entirely for simple queries (raw query used as-is)
       │
       ▼ fan-out (one branch per sub-query, parallel)
┌─────────────┐
│  Retrieve×N │  Dense (Qdrant) + BM25 (bm25s) → RRF fusion
└──────┬──────┘  All branches merge via operator.add on state
       │
       ▼
┌─────────────┐
│   Rerank    │  Dedup by chunk_id → Cohere cross-encoder (optional) → cap at 20
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Gate     │  Score threshold on Cohere relevance (0ms, no LLM)
└──────┬──────┘
       │
    ┌──┴────────────┐
    │               │
sufficient     insufficient (≤2 replans)
    │               │
    ▼               ▼
┌─────────┐   ┌─────────┐
│Synthesize│   │ Replan  │ → back to Planner
└─────────┘   └─────────┘
    │
    ▼
Cited answer + confidence score
```

**Model routing in synthesis:**
- `max_sub_queries ≤ 2` or `(==4, no targets)` → Haiku (~7s, ~$0.005/query)
- `max_sub_queries == 4` with target papers, or `== 5` → Sonnet (~14s, ~$0.015/query)

---

## Evaluation

Eval harness: RAGAS metrics + 11 custom type-specific metrics across 116 curated questions.

**Latest results (needy_sample tier, n=34):**

| Metric | Score |
|---|---|
| Faithfulness | 0.856 |
| Answer Relevancy | 0.770 |
| Context Recall | 0.909 |
| Context Precision | 0.854 |
| Comparison Score | 1.000 |
| Synthesis Score | 1.000 |
| Premise Score (false_premise) | 1.000 |
| Paper Coverage | 1.000 |
| Count Score (aggregation) | 1.000 |
| Contradiction Score | 0.700 |

**Eval tiers:**

| Tier | Questions | Use case |
|---|---|---|
| `fast` | 26 | Run after every change |
| `full` | 116 | Pre-release |
| `needy` | 75 | Cross-paper and adversarial types only |
| `sample` | 34 | Stratified 5/type sample — development iteration |

```bash
python -u -m eval.run_eval --tier fast
python -u -m eval.run_eval --tier sample
python -u -m eval.run_eval --tier full
```

---

## Ingestion

Seven-step pipeline that processes raw PDFs into indexed chunks:

1. **Parse** — LlamaParse converts PDF to markdown (cached; re-parsing is expensive and LlamaParse handles multi-column layouts better than open-source alternatives)
2. **Chunk** — Section-aware splitting at 512 tokens / 64 overlap. Tables are kept intact as single chunks. Figure/algorithm header artifacts are filtered before splitting.
3. **Metadata** — Regex extraction of section name, paper title, paper ID from chunk text. No LLM involved — fast and deterministic.
4. **Embed + store** — text-embedding-3-small → Qdrant (cosine, 1536 dims). Idempotent upsert with stable UUID5 point IDs.
5. **BM25 index** — bm25s keyword index over all chunks, persisted to `data/bm25/`. Rebuilt on corpus change.
6. **Entity extraction** — Haiku extracts named entities per chunk → Neo4j (opt-in, `--no-neo4j` to skip)
7. **Citation graph** — Paper citation edges → Neo4j (opt-in)

```bash
python -m ingestion.pipeline              # full pipeline
python -m ingestion.pipeline --no-neo4j  # skip graph steps (default in prod)
python -m ingestion.pipeline --dry-run   # parse + chunk only
```

---

## Retrieval

`TrimodalRetriever` fuses dense and sparse modalities via Reciprocal Rank Fusion:

```
score(d) = Σᵢ  1 / (k + rankᵢ(d))     k=60
```

- **Dense:** Qdrant vector search, top-20 candidates
- **BM25:** bm25s keyword search, top-20 candidates
- **Graph:** Neo4j entity traversal — built but disabled in production (`use_graph=False`). Dense+BM25 RRF already achieves 97% Hit@5; graph adds precision overhead without recall gains at this corpus size.

**Cohere reranking** is available but off by default. At 97% recall on a 10-paper corpus, the initial pool is clean enough that cross-encoder reranking reshuffles rather than improves results. It's re-evaluated as corpus size grows.

---

## Services

```bash
docker-compose up
```

| Service | Port | Purpose |
|---|---|---|
| `api` | 8000 | FastAPI — `/query` (JSON), `/query/stream` (SSE) |
| `frontend` | 3001 | Next.js 14 — chat, library, answer detail |
| `qdrant` | 6333 | Vector store |
| `langfuse` | 3000 | Observability — traces every node (latency, tokens, I/O) |

---

## Setup

```bash
# 1. Dependencies
pip install -e .

# 2. Environment
cp .env.example .env
# Fill: ANTHROPIC_API_KEY, OPENAI_API_KEY, COHERE_API_KEY (optional)

# 3. Drop PDFs in data/papers/ (max 10), then ingest
python -m ingestion.pipeline --no-neo4j

# 4. Test a query
python scripts/run_agent.py "What evaluation metrics does BEST-Route use?"

# 5. Run fast eval
python -u -m eval.run_eval --tier fast
```

---

## Repo structure

```
papermind/
├── api/                  FastAPI backend
├── query/
│   ├── agent.py          LangGraph graph — build_app(), run_agent()
│   ├── state.py          PaperMindState TypedDict + custom reducers
│   ├── retriever.py      TrimodalRetriever (dense + BM25 + optional graph)
│   └── nodes/
│       ├── classifier.py  Complexity routing → max_sub_queries, target_papers
│       ├── planner.py     Sub-query decomposition
│       ├── retrieval.py   Per-sub-query retrieve node (singleton retriever)
│       ├── rerank.py      Dedup + optional Cohere cross-encoder
│       ├── gate.py        Score-threshold confidence gate
│       ├── replan.py      Alternative query generation on low confidence
│       └── synthesis.py   Cited answer generation (Haiku or Sonnet)
├── ingestion/             7-step document processing pipeline
├── eval/
│   ├── run_eval.py        Eval harness (RAGAS + custom metrics)
│   ├── thresholds.py      Per-metric pass/fail thresholds
│   └── data/              Curated golden sets (fast/full/needy/sample tiers)
├── frontend/              Next.js 14 UI
├── scripts/               Dev utilities (run_agent, audit, chunk inspection)
└── docs/
    ├── DESIGN.md          RAGAS progression table (phase-by-phase)
    └── ENGINEERING.md     Engineering decisions with full reasoning
```

---

## Known limitations

- **10-paper corpus only.** The eval set, retrieval tuning, and synthesis prompts are calibrated for this corpus. Scaling requires re-tuning chunk size, RRF k, and rerank thresholds.
- **Contradiction detection is the weakest type** (0.70 score). When two papers hold opposing positions on a topic, retrieval needs to surface chunks from both sides — not guaranteed when one paper's vocabulary dominates the query embedding space.
- **Neo4j entity graph is stale.** Steps 6–7 were run on an earlier ingestion version. Re-run with `--no-neo4j` skipped; the graph component is not used in production retrieval.
- **Cohere reranking off by default.** Reevaluate when corpus grows beyond ~20 papers.
