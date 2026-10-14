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
st.title("RAG Knowledge Assistant")
st.caption(
    "Answers are generated only from your uploaded documents, with sources. "
    "Always check the cited passages: automated answers can still be wrong."
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
