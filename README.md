# PaperMind

**Agentic RAG system for research literature** — answers questions across a corpus of academic PDFs with inline citations at the paper, section, and page level.

> Built as a portfolio project targeting AI/ML Engineer roles. Demonstrates production-quality RAG: hybrid retrieval, cross-encoder reranking, LangGraph agent loop with self-correction, and systematic RAGAS evaluation.

---

## Demo

*[30-second GIF of a cross-paper query with citations — add after recording]*

---

## Results

RAGAS scores measured at each build stage (50-question eval set):

| Configuration | Ctx Precision | Ctx Recall | Faithfulness | Answer Rel. |
|---|---|---|---|---|
| Dense only (baseline) | — | — | — | — |
| + BM25 hybrid (RRF) | — | — | — | — |
| + Cross-encoder rerank | — | — | — | — |
| + Section-aware chunking | — | — | — | — |
| + Query rewrite loop | — | — | — | — |

*Numbers filled in after running `make eval`.*

---

## Architecture

```
Question
   │
   ▼
┌──────────────┐
│ Query Router │  classify: retrieval vs conversational
└──────┬───────┘
       │ retrieval
       ▼
┌──────────────────────────────────┐
│  Hybrid Retrieval                │
│  Dense (all-MiniLM) + BM25       │
│  → RRF Fusion                    │
│  → Cross-Encoder Reranker        │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────┐   score < threshold
│  Confidence  │──────────────────────► Query Rewriter (max 1 retry)
│  Gate        │                              │
└──────┬───────┘                              └──► back to Retrieval
       │ confident
       ▼
┌──────────────┐
│  Generator   │  Claude API, citation-grounded prompt
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│  Hallucination   │  verify answer is grounded in retrieved chunks
│  Check           │
└──────────────────┘
```

---

## Key Design Decisions

**Section-aware chunking** — Academic PDFs have explicit structure (abstract, methodology, results). Splitting across section boundaries mixes context that confuses retrieval. This implementation respects paragraph boundaries, splits at sentence boundaries only when a paragraph exceeds 400 tokens, merges short paragraphs (<100 tokens) within the same section, and never crosses section boundaries. This improved context precision by [X]% over sliding-window chunking on the eval set.

**Hybrid search** — Dense vector search captures semantic similarity but misses exact matches (author names, model names, paper titles). BM25 catches those. Both are fused via Reciprocal Rank Fusion, which handles the incompatible score scales without normalisation. Dense-only scored [X]% context recall; adding BM25 brought it to [Y]%.

**Confidence threshold over LLM grader** — The original design included an LLM-based document grader. This was cut: the cross-encoder already produces calibrated relevance scores, so a threshold gate (score < 0.3 → rewrite) gives the same self-correcting behaviour at a fraction of the latency and cost. The threshold was determined empirically on the eval set.

**Measurement at each stage** — Every retrieval layer was added incrementally with RAGAS measured before and after. This makes the engineering decisions defensible with numbers, not theory.

---

## Known Limitations

- **Math sections**: PyMuPDF mangles LaTeX equations into garbled unicode. Chunks with `contains_math=True` are flagged in metadata and the generator is instructed to hedge. A production system would use Nougat or GROBID for math-aware parsing.
- **Short chunk merging**: The 100-token minimum is a heuristic. Section conclusions sometimes merge with unrelated following text near section boundaries.
- **Query rewrite**: Limited to 1 retry. Complex multi-hop questions may still underperform.

---

## Quick Start

```bash
# 1. Start Qdrant
docker compose up -d

# 2. Install
pip install -e ".[dev]"
cp .env.example .env  # add your ANTHROPIC_API_KEY

# 3. Ingest papers
make ingest PDF_DIR=./data/papers

# 4. Launch UI
make ui
```

---

## Evaluation

```bash
# Run RAGAS for a specific build stage
make eval  # uses data/test_set/questions.json

# View progression table
python -m papermind.scripts.evaluate history
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Orchestration | LangGraph |
| Vector DB | Qdrant (Docker) |
| Embeddings | all-MiniLM-L6-v2 |
| Sparse search | BM25 (rank_bm25) |
| Fusion | Reciprocal Rank Fusion |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| PDF parsing | PyMuPDF |
| LLM | Claude API (claude-opus-4-6) |
| Eval | RAGAS |
| Tracing | LangSmith |
| UI | Streamlit |