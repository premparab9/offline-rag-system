
import os
import tempfile
from pathlib import Path

import streamlit as st

from ingest import ingest_document
from llm import AVAILABLE_MODELS
from query import answer_question
from vector_store import clear_collection, get_storage_stats

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DocIntel — Offline RAG",
    layout="wide",
    menu_items={},
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #f9fafb !important;
    color: #111827 !important;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

/* ── Sidebar ─────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e5e7eb !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] section { padding-top: 0 !important; }
[data-testid="stSidebar"] * { font-family: 'Inter', sans-serif !important; }

.sb-brand {
    padding: 20px 16px 16px;
    border-bottom: 1px solid #f3f4f6;
    margin-bottom: 4px;
}
.sb-brand-name {
    font-size: 15px;
    font-weight: 600;
    color: #111827;
    letter-spacing: -0.2px;
}
.sb-brand-tag {
    font-size: 11px;
    color: #9ca3af;
    margin-top: 2px;
}
.sb-section {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #9ca3af;
    padding: 16px 0 6px 0;
}
.sb-stat-row { display: flex; gap: 8px; margin: 6px 0 2px; }
.sb-stat {
    flex: 1;
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 10px 12px;
}
.sb-stat-val {
    font-size: 15px;
    font-weight: 600;
    color: #111827;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.2;
}
.sb-stat-lbl { font-size: 10px; color: #9ca3af; margin-top: 3px; }

/* ── Page header ─────────────────────────── */
.page-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 28px 0 20px;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 28px;
}
.page-title { font-size: 20px; font-weight: 600; color: #111827; letter-spacing: -0.3px; }
.page-sub   { font-size: 12px; color: #6b7280; margin-top: 4px; }
.offline-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    color: #15803d;
    font-size: 11px;
    font-weight: 500;
    padding: 5px 12px;
    border-radius: 20px;
}
.pulse-dot {
    width: 6px; height: 6px;
    background: #22c55e;
    border-radius: 50%;
    animation: blink 2s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }

/* ── Panel wrapper ───────────────────────── */
.panel {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 20px 20px 24px;
}
.panel-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #9ca3af;
    margin-bottom: 14px;
}

/* ── Answer card ─────────────────────────── */
.answer-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    overflow: hidden;
    margin-top: 20px;
}
.answer-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 11px 18px;
    background: #f9fafb;
    border-bottom: 1px solid #e5e7eb;
}
.answer-card-title {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #2563eb;
}
.answer-card-meta { display: flex; gap: 14px; }
.meta-item { font-size: 11px; color: #9ca3af; font-family: 'JetBrains Mono', monospace; }
.meta-item b { color: #374151; }
.answer-card-body {
    padding: 18px;
    font-size: 14px;
    line-height: 1.75;
    color: #374151;
}

/* ── Source chunks ───────────────────────── */
.chunk-card {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 8px;
}
.chunk-tags { display: flex; align-items: center; gap: 6px; margin-bottom: 10px; flex-wrap: wrap; }
.tag {
    font-size: 10px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.3px;
}
.tag-chunk  { background: #eff6ff; color: #2563eb; border: 1px solid #bfdbfe; }
.tag-page   { background: #f0fdf4; color: #16a34a; border: 1px solid #bbf7d0; }
.tag-bm25   { background: #faf5ff; color: #7c3aed; border: 1px solid #e9d5ff; }
.tag-file   { background: #f9fafb; color: #6b7280; border: 1px solid #e5e7eb; margin-left: auto; }
.chunk-text {
    font-size: 12px;
    line-height: 1.6;
    color: #6b7280;
    font-family: 'JetBrains Mono', monospace;
    border-top: 1px solid #e5e7eb;
    padding-top: 10px;
    white-space: pre-wrap;
    word-break: break-word;
}

/* ── Widget overrides ────────────────────── */
.stButton > button {
    background: #2563eb !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 0.45rem 1rem !important;
    transition: background .15s !important;
}
.stButton > button:hover  { background: #1d4ed8 !important; }

[data-testid="stTextArea"] textarea {
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    border-radius: 8px !important;
    border: 1px solid #d1d5db !important;
    background: #ffffff !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,.08) !important;
}
[data-testid="stSelectbox"] > div > div {
    border-radius: 8px !important;
    border: 1px solid #d1d5db !important;
    font-size: 13px !important;
}
[data-testid="stFileUploader"] {
    border-radius: 8px !important;
}
[data-testid="stExpander"] {
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    background: #ffffff !important;
}
[data-testid="stExpander"] summary p {
    font-size: 12px !important;
    color: #6b7280 !important;
    font-family: 'Inter', sans-serif !important;
}
.stProgress > div > div > div {
    background: linear-gradient(90deg, #2563eb, #60a5fa) !important;
    border-radius: 4px !important;
}
.stAlert { border-radius: 8px !important; font-size: 13px !important; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-brand-name">DocIntel</div>
        <div class="sb-brand-tag">Offline RAG Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Inference Model</div>', unsafe_allow_html=True)
    model_keys   = list(AVAILABLE_MODELS.keys())
    model_labels = [AVAILABLE_MODELS[k]["label"] for k in model_keys]
    chosen_label = st.selectbox("Model", model_labels, index=0, label_visibility="collapsed")
    chosen_model = model_keys[model_labels.index(chosen_label)]
    st.caption(AVAILABLE_MODELS[chosen_model]["description"])

    st.markdown('<div class="sb-section">Vector Store</div>', unsafe_allow_html=True)
    stats = get_storage_stats()
    st.markdown(f"""
    <div class="sb-stat-row">
        <div class="sb-stat">
            <div class="sb-stat-val">{stats["size_readable"]}</div>
            <div class="sb-stat-lbl">Index Size</div>
        </div>
        <div class="sb-stat">
            <div class="sb-stat-val">{stats["chunk_count"]}</div>
            <div class="sb-stat-lbl">Chunks</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Actions</div>', unsafe_allow_html=True)
    if st.button("Clear Knowledge Base", use_container_width=True):
        clear_collection()
        st.success("Knowledge base cleared.")
        st.rerun()

st.markdown("""
<div class="page-header">
    <div>
        <div class="page-title">Document Intelligence</div>
        <div class="page-sub">Private document Q&amp;A — all processing happens locally, no data leaves this machine</div>
    </div>
    <div class="offline-badge">
        <div class="pulse-dot"></div>
        Offline &amp; Secure
    </div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="panel-label">Document Ingestion</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        names = [f.name for f in uploaded]
        st.caption(f"{len(uploaded)} file(s) selected: " + ", ".join(names[:3]) +
                   (f" +{len(names)-3} more" if len(names) > 3 else ""))

        if st.button("Ingest Documents", type="primary", use_container_width=True):
            bar   = st.progress(0)
            log_  = []

            for i, file in enumerate(uploaded):
                suffix = Path(file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(file.getbuffer())
                    tmp_path = tmp.name

                with st.spinner(f"Processing {file.name}..."):
                    n = ingest_document(tmp_path)

                os.unlink(tmp_path)
                bar.progress((i + 1) / len(uploaded))
                log_.append((file.name, n))

            st.success(f"Ingestion complete — {len(uploaded)} file(s) processed.")
            for fname, n in log_:
                label = f"{n} chunks stored" if n else "no text extracted"
                st.caption(f"{fname}  —  {label}")

            st.rerun()

with right:
    st.markdown('<div class="panel-label">Query</div>', unsafe_allow_html=True)

    question = st.text_area(
        "Question",
        placeholder="Ask a question about your documents...",
        height=130,
        label_visibility="collapsed",
    )

    q_col, s_col = st.columns([3, 1])
    with q_col:
        run = st.button("Run Query", type="primary", use_container_width=True)
    with s_col:
        show_src = st.checkbox("Sources", value=True)

    if run:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving context and generating answer..."):
                result = answer_question(question, model_name=chosen_model)

            st.markdown(f"""
            <div class="answer-card">
                <div class="answer-card-header">
                    <span class="answer-card-title">Answer</span>
                    <div class="answer-card-meta">
                        <span class="meta-item"><b>{result["routed_model"]}</b></span>
                        <span class="meta-item"><b>{result["time_taken"]}s</b></span>
                        <span class="meta-item"><b>{len(result["sources"])}</b> source(s)</span>
                    </div>
                </div>
                <div class="answer-card-body">{result["answer"]}</div>
            </div>
            """, unsafe_allow_html=True)

            if show_src and result["sources"]:
                with st.expander(f"Retrieved context  —  {len(result['sources'])} chunk(s)"):
                    for i, chunk in enumerate(result["sources"], 1):
                        score = chunk.get("bm25_score", 0.0)
                        text  = chunk["text"][:420].strip().replace("<", "&lt;").replace(">", "&gt;")
                        st.markdown(f"""
                        <div class="chunk-card">
                            <div class="chunk-tags">
                                <span class="tag tag-chunk">CHUNK {i}</span>
                                <span class="tag tag-page">PAGE {chunk["page"]}</span>
                                <span class="tag tag-bm25">BM25 {score:.4f}</span>
                                <span class="tag tag-file">{chunk["source"]}</span>
                            </div>
                            <div class="chunk-text">{text}...</div>
                        </div>
                        """, unsafe_allow_html=True)
