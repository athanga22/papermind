"""
PaperMind Streamlit UI.

Run with:
    streamlit run src/papermind/ui/app.py
"""

from __future__ import annotations

import streamlit as st

from ..agent.graph import run as agent_run
from ..ingestion.models import RetrievedChunk

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PaperMind",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📄 PaperMind")
    st.caption("Agentic RAG for Research Literature")
    st.divider()
    st.markdown(
        """
        **How it works**
        1. Upload PDFs via `make ingest PDF_DIR=./data/papers`
        2. Ask questions about your paper corpus
        3. Answers cite [Paper, Section, Page] for every claim

        **Agent loop**
        - Query routing → dense + BM25 hybrid search
        - RRF fusion → cross-encoder reranking
        - Confidence gate → query rewriting if needed
        - Citation-grounded generation → hallucination check
        """
    )
    st.divider()
    show_debug = st.checkbox("Show retrieval debug info", value=False)


# ── Main chat interface ───────────────────────────────────────────────────────

st.title("Ask about your papers")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations") and show_debug:
            _render_citations(msg["citations"])

# Input
if prompt := st.chat_input("Ask a question across your research papers…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Searching and reasoning…"):
            state = agent_run(prompt)

        answer = state.get("answer", "No answer generated.")
        citations: list[dict] = state.get("citations", [])  # type: ignore[assignment]
        is_grounded = state.get("is_grounded", True)
        issues = state.get("grounding_issues", [])
        retry_count = state.get("retry_count", 0)

        st.markdown(answer)

        # Grounding warning
        if not is_grounded and issues:
            with st.expander("⚠️ Grounding issues detected", expanded=True):
                for issue in issues:
                    st.warning(issue)

        # Query rewrite indicator
        if retry_count > 0:
            st.caption("🔄 Query was rewritten to improve retrieval.")

        # Citation chain
        if citations and show_debug:
            _render_citations(citations)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "citations": citations}
    )


def _render_citations(citations: list[dict]) -> None:  # type: ignore[type-arg]
    """Render the retrieval chain — shown only when debug mode is on."""
    with st.expander("📎 Retrieved context", expanded=False):
        for i, c in enumerate(citations, start=1):
            score_bar = "█" * int(c.get("score", 0) * 10) + "░" * (10 - int(c.get("score", 0) * 10))
            st.markdown(
                f"**[{i}] {c['paper_title']}** · {c['section'].title()} · p.{c['page_number']}  \n"
                f"`score: {c.get('score', 0):.3f}` `{score_bar}`  \n"
                f"_{c.get('text_preview', '')}…_"
            )
            if i < len(citations):
                st.divider()
