"""Streamlit user interface.

Run:  streamlit run frontend/streamlit_app.py
All work happens in the FastAPI backend (API_URL); this file only renders
results and forwards user actions.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api_client import ApiClient, BackendError

API_URL = os.getenv("API_URL", "http://localhost:8000")
MIME_TYPES = {"pdf": "application/pdf", "txt": "text/plain"}

st.set_page_config(page_title="RAG Knowledge Assistant", page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
    :root { --kp-ink: #172033; --kp-muted: #64748b; --kp-line: #e6eaf2; --kp-purple: #625bf6; }
    .stApp { background: linear-gradient(180deg, #f7f8ff 0%, #fbfcff 240px, #f8fafc 100%); }
    [data-testid="stHeader"] { background: rgba(255,255,255,.72); }
    [data-testid="stSidebar"] { background: #fff; border-right: 1px solid var(--kp-line); }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.25rem; }
    h1 { color: var(--kp-ink); letter-spacing: -1.4px; font-weight: 800 !important; }
    h2, h3 { color: var(--kp-ink); letter-spacing: -.4px; }
    [data-testid="stChatMessage"] { border: 1px solid var(--kp-line); border-radius: 16px; padding: 1rem 1.1rem; background: #fff; box-shadow: 0 5px 18px rgba(26,35,66,.035); }
    [data-testid="stChatInput"] { border-radius: 16px; border-color: #dfe4f1; box-shadow: 0 8px 28px rgba(36,45,83,.06); background: #fff; }
    [data-testid="stFileUploader"] { border: 1px dashed #b8b9f5; border-radius: 14px; background: #fafaff; padding: .35rem; }
    .kp-eyebrow { color: var(--kp-purple); text-transform: uppercase; font-size: .72rem; letter-spacing: .14em; font-weight: 800; margin-bottom: .35rem; }
    .kp-subtitle { color: var(--kp-muted); font-size: 1rem; line-height: 1.65; max-width: 760px; margin-bottom: 1.2rem; }
    .kp-stat { background: #fff; border: 1px solid var(--kp-line); border-radius: 15px; padding: 1rem 1.1rem; min-height: 100px; box-shadow: 0 5px 18px rgba(26,35,66,.035); }
    .kp-stat-label { color: var(--kp-muted); font-size: .78rem; font-weight: 650; margin-bottom: .45rem; }
    .kp-stat-value { color: var(--kp-ink); font-size: 1.65rem; font-weight: 800; letter-spacing: -.05em; }
    .kp-stat-note { color: #7c86a0; font-size: .72rem; margin-top: .2rem; }
    div.stButton > button[kind="primary"] { border-radius: 10px; font-weight: 700; border: 0; box-shadow: 0 5px 14px rgba(98,91,246,.18); }
    div[data-testid="stExpander"] { border-radius: 12px; border-color: var(--kp-line); background: #fff; }
    @media (max-width: 700px) { .kp-stat { padding: .8rem; min-height: 90px; } .kp-stat-value { font-size: 1.35rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)
client = ApiClient(API_URL)

ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("uploader_key", 0)
ss.setdefault("flash", [])


def flash(kind: str, text: str) -> None:
    """Queue a message that survives the next st.rerun()."""
    ss.flash.append((kind, text))


@st.cache_data(ttl=10, show_spinner=False)
def fetch_health() -> dict[str, Any] | None:
    try:
        return client.health()
    except BackendError:
        return None


def fetch_documents() -> list[dict[str, Any]]:
    try:
        return client.list_documents()
    except BackendError as exc:
        st.sidebar.error(exc.message)
        return []


def upload_files(files: list[Any]) -> None:
    for f in files:
        ext = f.name.rsplit(".", 1)[-1].lower()
        with st.spinner(f"Indexing {f.name}…"):
            try:
                doc = client.upload(f.name, f.getvalue(), MIME_TYPES.get(ext, "text/plain"))
            except BackendError as exc:
                if exc.status_code == 409:
                    flash("info", f"**{f.name}**: {exc.message}")
                else:
                    flash("error", f"**{f.name}**: {exc.message}")
                continue
        flash(
            "success",
            f"Indexed **{doc['filename']}**: {doc['page_count']} page(s), "
            f"{doc['chunk_count']} chunk(s).",
        )
        for warning in doc.get("warnings", []):
            flash("warning", f"**{doc['filename']}**: {warning}")
    ss.uploader_key += 1  # reset the file picker
    fetch_health.clear()
    st.rerun()


def delete_document(doc: dict[str, Any]) -> None:
    try:
        client.delete(doc["document_id"])
        flash("success", f"Deleted **{doc['filename']}** and its index entries.")
    except BackendError as exc:
        flash("error", exc.message)
    fetch_health.clear()
    st.rerun()


def render_citations(citations: list[dict[str, Any]]) -> None:
    if not citations:
        return
    explicit = citations[0].get("explicitly_cited", True)
    st.markdown("**Sources**" if explicit else "**Retrieved passages** *(not cited explicitly)*")
    for c in citations:
        where = c["filename"] + (f" — page {c['page_number']}" if c.get("page_number") else "")
        with st.expander(f"[{c['label']}] {where}  ·  relevance {c['score']:.2f}"):
            st.text(c["excerpt"])  # plain text: document content is never rendered as markup
            st.caption(f"Chunk `{c['chunk_id']}`")


def render_assistant(msg: dict[str, Any]) -> None:
    if msg.get("error"):
        st.error(msg["error"])
        return
    if msg["status"] == "insufficient_evidence":
        st.info(msg["answer"], icon="🔍")
    else:
        st.markdown(msg["answer"])
    render_citations(msg.get("citations", []))
    if msg["provider"] == "extractive" and msg["status"] == "answered":
        st.caption("Extractive mode: these are quoted passages; no language model was used.")
    st.caption(f"{msg['provider']} · {msg['model']} · {msg['latency_ms']} ms")


# ---------------------------------------------------------------- sidebar --
health = fetch_health()
with st.sidebar:
    st.header("📚 Documents")
    for kind, text in ss.flash:
        getattr(st, kind)(text)
    ss.flash = []

    if health is None:
        st.error(f"Backend not reachable at {API_URL}. Start the API and refresh.")
    files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key=f"uploader_{ss.uploader_key}",
        help="Text-based PDFs only; scanned PDFs need OCR, which is not supported.",
        disabled=health is None,
    )
    if files and st.button("Upload & index", type="primary", width="stretch"):
        upload_files(files)

    documents = fetch_documents() if health else []
    if not documents and health:
        st.caption("No documents yet. Upload one to get started.")
    for doc in documents:
        col_name, col_btn = st.columns([5, 1], vertical_alignment="center")
        pages = "page" if doc["page_count"] == 1 else "pages"
        col_name.text(doc["filename"])  # plain text: never render user filenames as markup
        col_name.caption(f"{doc['page_count']} {pages} · {doc['chunk_count']} chunks")
        with col_btn.popover("🗑", help="Delete document"):
            st.write("Delete this document and its index entries?")
            if st.button("Delete", key=f"del_{doc['document_id']}", type="primary"):
                delete_document(doc)

    st.divider()
    st.header("⚙️ Retrieval settings")
    top_k = st.slider("Passages to retrieve (top-k)", 1, 10, 5)
    custom_threshold = st.toggle(
        "Custom relevance threshold",
        value=False,
        help="Off = use the server default for the embedding model.",
    )
    min_score = (
        st.slider(
            "Minimum relevance score",
            0.0,
            1.0,
            0.30,
            0.01,
            help="Passages below this similarity are ignored. A heuristic only: "
            "it reduces, but cannot prevent, unsupported answers.",
        )
        if custom_threshold
        else None
    )
    names = {d["document_id"]: d["filename"] for d in documents}
    selected = st.multiselect(
        "Search only these documents",
        options=list(names),
        format_func=lambda i: names[i],
        placeholder="All documents",
    )
    if st.button("Clear chat", width="stretch"):
        ss.messages = []
        st.rerun()

    st.divider()
    if health:
        llm = health["llm"]
        st.caption(f"LLM: **{health['llm_provider']}** — {llm['detail']}")
        st.caption(f"Embeddings: {health['embedding']['detail']}")
        if not llm["ok"]:
            st.warning("The answer model is not ready; see the README to configure it.")
        if health.get("llm_is_external"):
            st.warning("Questions and retrieved excerpts are sent to an external LLM service.")

# ------------------------------------------------------------------- main --
st.markdown('<div class="kp-eyebrow">Your private knowledge workspace</div>', unsafe_allow_html=True)
st.title("Ask your documents. Get grounded answers.")
st.markdown(
    '<div class="kp-subtitle">Search across your PDFs and text files, explore the exact passages behind each answer, '
    'and keep every response traceable to its source.</div>',
    unsafe_allow_html=True,
)

doc_total = len(documents) if health else 0
indexed_chunks = sum(int(d.get("chunk_count", 0)) for d in documents) if health else 0
status_label = "Connected" if health else "Offline"
status_color = "#0f9f78" if health else "#d97706"
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(
        f'<div class="kp-stat"><div class="kp-stat-label">📚 Indexed documents</div>'
        f'<div class="kp-stat-value">{doc_total}</div><div class="kp-stat-note">Available in this workspace</div></div>',
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        f'<div class="kp-stat"><div class="kp-stat-label">🧩 Searchable passages</div>'
        f'<div class="kp-stat-value">{indexed_chunks:,}</div><div class="kp-stat-note">Chunks available for retrieval</div></div>',
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        f'<div class="kp-stat"><div class="kp-stat-label">● Backend status</div>'
        f'<div class="kp-stat-value" style="font-size:1.45rem;color:{status_color}">{status_label}</div>'
        f'<div class="kp-stat-note">Live status from the API health check</div></div>',
        unsafe_allow_html=True,
    )
st.write("")
st.caption(
    "Answers are grounded in retrieved passages, but automated responses can still be wrong. "
    "Open the citations and verify the original text before relying on important information."
)

for msg in ss.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_assistant(msg)

if question := st.chat_input(
    "Ask a question about your documents", disabled=health is None or not documents
):
    ss.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and writing an answer…"):
            try:
                result: dict[str, Any] = client.ask(question, top_k, min_score, selected or None)
            except BackendError as exc:
                result = {"error": exc.message}
        result["role"] = "assistant"
        render_assistant(result)
    ss.messages.append(result)

if health and not documents and not ss.messages:
    st.info("👈 Upload a document in the sidebar, then ask a question here.")
