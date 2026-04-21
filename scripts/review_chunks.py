"""
Chunk Review UI — Streamlit app for verifying auto-populated chunk assignments.

For each auto_populated entry:
  - Shows question + reference answer
  - "🔍 Get Chunks" button retrieves top-5 live from the retriever
  - Each result has a "✅ Use this" button to assign it
  - Approve (keep current) / Reject (clear) always available

Progress saved to JSON after every action.

Usage:
    .venv/bin/streamlit run scripts/review_chunks.py
    .venv/bin/streamlit run scripts/review_chunks.py -- --file eval/data/my_set.json
"""

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=Path, default=ROOT / "eval/data/sample.json")
try:
    args = parser.parse_args()
except SystemExit:
    args = argparse.Namespace(file=ROOT / "eval/data/sample.json")

JSON_PATH = args.file

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_items():
    with open(JSON_PATH) as f:
        return json.load(f)

def save_items(items):
    with open(JSON_PATH, "w") as f:
        json.dump(items, f, indent=2)

@st.cache_resource
def get_retriever():
    from query.retriever import TrimodalRetriever
    return TrimodalRetriever()

def retrieve(question: str, paper_id: str | None = None):
    r = get_retriever()
    chunks = r.retrieve(question, top_k=5)
    if paper_id:
        chunks = [c for c in chunks if c.paper_id == paper_id]
        if not chunks:
            # fallback: retrieve more and filter
            chunks = r.retrieve(question, top_k=20)
            chunks = [c for c in chunks if c.paper_id == paper_id][:5]
    return chunks

@st.cache_resource
def get_all_papers():
    """Return {paper_id: paper_title} for all papers in Qdrant."""
    import os
    from qdrant_client import QdrantClient
    qdrant = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
    )
    results, _ = qdrant.scroll(
        collection_name="papers",
        limit=500,
        with_payload=["paper_id", "paper_title"],
        with_vectors=False,
    )
    papers = {}
    for pt in results:
        pid   = pt.payload["paper_id"]
        title = pt.payload["paper_title"]
        if pid not in papers:
            papers[pid] = title
    return dict(sorted(papers.items(), key=lambda x: x[1]))

# ── Session state ─────────────────────────────────────────────────────────────

if "questions" not in st.session_state:
    st.session_state.questions = load_items()
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "retrieved" not in st.session_state:
    st.session_state.retrieved = None
if "filter_paper" not in st.session_state:
    st.session_state.filter_paper = None

def review_queue(items):
    return [
        (i, item) for i, item in enumerate(items)
        if item.get("auto_populated") and not item.get("reviewed")
    ]

# ── Page ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Chunk Review", layout="wide")
st.title("🔍 Chunk Review")

items     = st.session_state.questions
queue     = review_queue(items)
all_papers = get_all_papers()

if not queue:
    st.success("🎉 All auto-populated entries reviewed!")
    approved = sum(1 for it in items if it.get("reviewed") and it.get("chunk_id"))
    rejected = sum(1 for it in items if it.get("auto_populated") and not it.get("chunk_id"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Total", len(items))
    c2.metric("Approved", approved)
    c3.metric("Rejected", rejected)
    st.stop()

# Progress
total_auto = sum(1 for it in items if it.get("auto_populated"))
done_auto  = total_auto - len(queue)
st.progress(done_auto / total_auto, text=f"{done_auto} / {total_auto} reviewed")

# Clamp
if st.session_state.idx >= len(queue):
    st.session_state.idx = 0

orig_idx, item = queue[st.session_state.idx]

# ── Navigation ────────────────────────────────────────────────────────────────

c_prev, c_mid, c_next = st.columns([1, 4, 1])
with c_prev:
    if st.button("← Prev") and st.session_state.idx > 0:
        st.session_state.idx -= 1
        st.session_state.retrieved = None
        st.rerun()
with c_mid:
    st.markdown(
        f"<div style='text-align:center;padding-top:6px'>"
        f"<b>{st.session_state.idx + 1}</b> of <b>{len(queue)}</b> pending</div>",
        unsafe_allow_html=True,
    )
with c_next:
    if st.button("Next →") and st.session_state.idx < len(queue) - 1:
        st.session_state.idx += 1
        st.session_state.retrieved = None
        st.rerun()

st.divider()

# ── Badge ─────────────────────────────────────────────────────────────────────

badge_color = {
    "factoid":       "#1f77b4",
    "limitation":    "#ff7f0e",
    "false_premise": "#d62728",
    "contradiction": "#9467bd",
}.get(item.get("type", ""), "#555")

st.markdown(
    f"<span style='background:{badge_color};color:white;padding:2px 10px;"
    f"border-radius:12px;font-size:0.85em'>{item.get('type','?').upper()}</span>"
    f"&nbsp;&nbsp;<span style='color:#888;font-size:0.85em'>{item.get('source','?')}</span>",
    unsafe_allow_html=True,
)
st.markdown("")

# ── Question + Answer ─────────────────────────────────────────────────────────

st.subheader("Question")
st.write(item["question"])

st.subheader("Reference Answer")
st.info(item.get("answer", "—"))

st.divider()

# ── Currently assigned ────────────────────────────────────────────────────────

current_cids = item.get("chunk_id") or []
if current_cids:
    st.markdown(f"**Currently assigned:** `{'`, `'.join(current_cids)}`")

# ── Verdict buttons ───────────────────────────────────────────────────────────

a1, a2 = st.columns(2)
with a1:
    if st.button("✅ Approve current assignment", use_container_width=True, type="primary"):
        st.session_state.questions[orig_idx]["reviewed"] = True
        save_items(st.session_state.questions)
        st.session_state.retrieved = None
        if st.session_state.idx < len(queue) - 1:
            st.session_state.idx += 1
        st.rerun()
with a2:
    if st.button("❌ Reject (clear chunk)", use_container_width=True):
        st.session_state.questions[orig_idx]["chunk_id"]  = []
        st.session_state.questions[orig_idx]["paper_ids"] = []
        st.session_state.questions[orig_idx]["reviewed"]  = False
        st.session_state.questions[orig_idx].pop("auto_populated", None)
        save_items(st.session_state.questions)
        st.session_state.retrieved = None
        st.rerun()

st.divider()

# ── Retrieval controls ────────────────────────────────────────────────────────

paper_options = {"(any paper)": None} | {f"{title[:60]}  [{pid}]": pid for pid, title in all_papers.items()}
selected_label = st.selectbox("Filter retrieval to paper:", list(paper_options.keys()))
selected_paper = paper_options[selected_label]

if st.button("🔍 Get Chunks", use_container_width=True):
    with st.spinner("Retrieving..."):
        st.session_state.retrieved = retrieve(item["question"], paper_id=selected_paper)

# ── Results ───────────────────────────────────────────────────────────────────

if st.session_state.retrieved is not None:
    chunks = st.session_state.retrieved
    if not chunks:
        st.warning("No chunks found — try a different paper filter or check if the paper is ingested.")
    else:
        st.subheader(f"Top {len(chunks)} Retrieved Chunks")
        for rank, chunk in enumerate(chunks, 1):
            with st.container(border=True):
                col_info, col_btn = st.columns([5, 1])
                with col_info:
                    st.markdown(
                        f"**[{rank}] `{chunk.chunk_id}`** &nbsp;|&nbsp; "
                        f"*{chunk.paper_title[:60]}*  \n"
                        f"Section: `{chunk.section}` &nbsp;|&nbsp; "
                        f"Score: `{chunk.score:.4f}` &nbsp;|&nbsp; "
                        f"Sources: `{chunk.sources}`"
                    )
                    st.markdown(
                        f"<div style='background:#1e1e1e;color:#e0e0e0;border:1px solid #444;"
                        f"padding:12px;border-radius:6px;font-size:0.9em;white-space:pre-wrap'>"
                        f"{chunk.text[:700]}</div>",
                        unsafe_allow_html=True,
                    )
                with col_btn:
                    st.markdown("<div style='padding-top:28px'></div>", unsafe_allow_html=True)
                    if st.button("✅ Use this", key=f"use_{rank}_{chunk.chunk_id}"):
                        q = st.session_state.questions[orig_idx]
                        if item.get("type") == "contradiction":
                            existing        = q.get("chunk_id") or []
                            existing_papers = q.get("paper_ids") or []
                            if chunk.chunk_id not in existing:
                                existing.append(chunk.chunk_id)
                                existing_papers.append(chunk.paper_id)
                            q["chunk_id"]  = existing
                            q["paper_ids"] = existing_papers
                        else:
                            q["chunk_id"]  = [chunk.chunk_id]
                            q["paper_ids"] = [chunk.paper_id]
                        q["auto_populated"] = True
                        save_items(st.session_state.questions)
                        st.rerun()

# ── Raw JSON ──────────────────────────────────────────────────────────────────

with st.expander("Raw JSON entry"):
    st.json(item)
