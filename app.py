import streamlit as st
import tempfile
import os
from pathlib import Path

from ingest import ingest_document
from query import answer_question
from vector_store import clear_collection
from config import LLM_MODEL

# Page configuration
st.set_page_config(
    page_title="Offline RAG System",
    page_icon="🔒",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""

""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────
st.title("🔒 Offline RAG System")
st.markdown("*Ask questions about your documents — 100% private, no internet.*")
st.divider()

# ── Sidebar ───────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    # Model selector — lists models pulled via Ollama
    selected_model = st.selectbox(
        "Select LLM",
        options=["gemma2:2b", "gemma2:9b", "deepseek-r1:7b", "llama3.2:3b"],
        index=0
    )
    st.caption(f"Active model: **{selected_model}**")

    st.divider()

    # Knowledge base management
    st.subheader("📚 Knowledge Base")
    if st.button("🗑️ Clear All Documents", use_container_width=True):
        clear_collection()
        st.success("Knowledge base cleared!")

# ── Two-column layout ─────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

# ── Column 1: Document Upload ─────────────────────────
with col1:
    st.subheader("📄 Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("⚡ Ingest Documents", type="primary",
                     use_container_width=True):
            progress = st.progress(0)

            for i, file in enumerate(uploaded_files):
                with st.spinner(f"Processing {file.name}..."):
                    # Save to temp file (Streamlit gives us bytes)
                    suffix = Path(file.name).suffix
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix
                    ) as tmp:
                        tmp.write(file.getbuffer())
                        tmp_path = tmp.name

                    # Ingest the document
                    chunks_added = ingest_document(tmp_path)
                    os.unlink(tmp_path)   # Clean up temp file

                    progress.progress((i + 1) / len(uploaded_files))
                    st.success(f"✓ {file.name} — {chunks_added} chunks stored")

# ── Column 2: Q&A ─────────────────────────────────────
with col2:
    st.subheader("💬 Ask a Question")

    question = st.text_area(
        "Your question",
        placeholder="What does the document say about...?",
        height=120
    )

    show_sources = st.checkbox("Show source chunks", value=True)

    if st.button("🔍 Get Answer", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                result = answer_question(question, model_name=selected_model)

            st.markdown("### Answer")
            st.markdown(result["answer"])

            if show_sources and result["sources"]:
                with st.expander("📌 Source Chunks Used"):
                  for i, chunk in enumerate(result["sources"], 1):
                     st.markdown(f"**Chunk {i}:** {chunk[:300]}...")