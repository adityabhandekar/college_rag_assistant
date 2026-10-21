"""
app.py
Streamlit frontend for the College RAG Assistant.
Connects to the actual backend. No hardcoded answers.
"""

import logging
import os

import streamlit as st

from config import (
    DATA_DIR, CHAT_MODEL, EMBEDDING_MODEL, UNKNOWN_ANSWER,
    TOP_K, SIMILARITY_THRESHOLD
)
from document_loader import load_all_documents
from chunker import chunk_pages
from vector_store import (
    clear_collection, add_chunks, get_collection_count, get_all_sources
)
from rag_service import ask_question

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="College Document Q&A Assistant",
    page_icon="🎓",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar: Document management
# ---------------------------------------------------------------------------
st.sidebar.title("📚 Document Management")

pdf_files = []
if os.path.isdir(DATA_DIR):
    pdf_files = [f for f in sorted(os.listdir(DATA_DIR)) if f.lower().endswith(".pdf")]

st.sidebar.markdown(f"**Data folder:** `{DATA_DIR}`")
st.sidebar.markdown(f"**PDF files found:** {len(pdf_files)}")

if pdf_files:
    for pdf in pdf_files:
        st.sidebar.markdown(f"- {pdf}")
else:
    st.sidebar.warning("No PDF files found. Add PDFs to the data/ folder.")

st.sidebar.markdown("---")

# Index status
try:
    chunk_count = get_collection_count()
    sources = get_all_sources()
except Exception:
    chunk_count = 0
    sources = []

st.sidebar.markdown(f"**Indexed chunks:** {chunk_count}")
if sources:
    st.sidebar.markdown("**Indexed documents:**")
    for s in sources:
        st.sidebar.markdown(f"- {s}")
else:
    st.sidebar.info("No documents indexed yet.")

st.sidebar.markdown("---")

# Re-index button
if st.sidebar.button("🔄 Re-index Documents", use_container_width=True):
    if not pdf_files:
        st.sidebar.error("No PDF files to index.")
    else:
        with st.spinner("Clearing old index and re-indexing..."):
            try:
                clear_collection()
                pages = load_all_documents(DATA_DIR)
                if not pages:
                    st.sidebar.error("No text could be extracted from the PDFs.")
                else:
                    chunks = chunk_pages(pages)
                    added = add_chunks(chunks)
                    st.sidebar.success(f"Indexed {added} chunks from {len(pages)} pages.")
                    st.rerun()
            except RuntimeError as exc:
                st.sidebar.error(str(exc))
            except Exception as exc:
                st.sidebar.error(f"Unexpected error during indexing: {exc}")

st.sidebar.markdown("---")
st.sidebar.caption(
    f"**Models:** `{CHAT_MODEL}` (chat), `{EMBEDDING_MODEL}` (embeddings)\n\n"
    f"**Retrieval:** top-{TOP_K}, distance threshold ≤ {SIMILARITY_THRESHOLD}\n\n"
    "All processing is local. No data leaves your machine."
)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("🎓 College Document Q&A Assistant")
st.markdown(
    "Ask questions about college documents. The assistant answers **only** from the "
    "indexed documents and refuses to invent information."
)

# System info expander
with st.expander("ℹ️ About / System Information"):
    st.markdown(
        f"""
        **Architecture:** Local Retrieval-Augmented Generation (RAG)

        - **Document ingestion:** PyPDF extracts text page by page.
        - **Chunking:** ~{1500} characters with {200} characters overlap.
        - **Embeddings:** Ollama `{EMBEDDING_MODEL}` converts chunks and queries to vectors.
        - **Vector database:** ChromaDB stores embeddings persistently on disk.
        - **Retrieval:** Top-{TOP_K} chunks are retrieved by cosine similarity.
          Chunks with distance > {SIMILARITY_THRESHOLD} are discarded.
        - **Generation:** Ollama `{CHAT_MODEL}` generates an answer from the retrieved context only.
        - **Refusal:** If no sufficient context is found, the assistant replies:
          "{UNKNOWN_ANSWER}"

        **Privacy:** Everything runs on your computer. No API keys are required.
        """
    )

# Question input
question = st.text_input(
    "Enter your question:",
    placeholder="e.g., What is the minimum attendance requirement?",
)

col1, col2 = st.columns([1, 5])
with col1:
    ask_clicked = st.button("Ask", type="primary", use_container_width=True)
with col2:
    if st.button("Clear", use_container_width=True):
        st.rerun()

# ---------------------------------------------------------------------------
# Handle ask
# ---------------------------------------------------------------------------
if ask_clicked:
    if not question or not question.strip():
        st.warning("Please enter a question.")
    elif chunk_count == 0:
        st.error(
            "No documents are indexed. Add PDF files to the `data/` folder "
            "and click **Re-index Documents** in the sidebar."
        )
    else:
        with st.spinner("Retrieving relevant information and generating answer..."):
            try:
                result = ask_question(question.strip())
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")
                result = None

        if result:
            # Answer
            st.markdown("### Answer")
            if result["refused"]:
                st.warning(result["answer"])
            else:
                st.success(result["answer"])

            # Sources
            if result["sources"]:
                st.markdown("### Sources")
                for i, src in enumerate(result["sources"], start=1):
                    st.markdown(
                        f"**{i}. {src['source']}** — Page {src['page']} "
                        f"(distance: {src['distance']:.4f})"
                    )

            # Retrieved excerpts
            if result["retrieved"]:
                with st.expander("🔍 View Retrieved Excerpts"):
                    for i, chunk in enumerate(result["retrieved"], start=1):
                        st.markdown(
                            f"**Chunk {i}** — {chunk['source']}, Page {chunk['page']} "
                            f"(distance: {chunk['distance']:.4f})"
                        )
                        st.text_area(
                            label=f"Excerpt {i}",
                            value=chunk["text"][:800] + ("..." if len(chunk["text"]) > 800 else ""),
                            height=120,
                            label_visibility="collapsed",
                            key=f"chunk_{i}",
                        )
            elif not result["refused"]:
                st.info("No excerpts passed the relevance threshold.")