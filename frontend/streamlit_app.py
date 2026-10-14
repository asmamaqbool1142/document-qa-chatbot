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
    :root { --kp-ink: #20213a; --kp-muted: #737792; --kp-line: #e7e7f5; --kp-purple: #6657f5; --kp-blue: #4388f5; --kp-pink: #f16da8; --kp-teal: #13b8a6; }
    .stApp { background: radial-gradient(ellipse at 10% 0%, #f0edff 0, transparent 35%), radial-gradient(ellipse at 95% 5%, #e5f5ff 0, transparent 30%), linear-gradient(180deg, #fbfaff 0%, #f7f8ff 100%); }
    [data-testid="stHeader"] { background: rgba(255,255,255,.75); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg,#fff 0%,#f6f5ff 100%); border-right: 1px solid #e4e2fa; }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.25rem; }
    h1 { color: var(--kp-ink); letter-spacing: -1.4px; font-weight: 850 !important; }
    h2, h3 { color: var(--kp-ink); letter-spacing: -.4px; }
    [data-testid="stChatMessage"] { border: 1px solid #e8e5fa; border-radius: 18px; padding: 1rem 1.1rem; background: rgba(255,255,255,.94); box-shadow: 0 8px 24px rgba(66,56,145,.07); }
    [data-testid="stChatInput"] { border-radius: 16px; border-color: #d9d5fb; box-shadow: 0 8px 28px rgba(80,65,180,.10); background: #fff; }
    [data-testid="stFileUploader"] { border: 1.5px dashed #a9a0fa; border-radius: 16px; background: linear-gradient(135deg,#f8f6ff,#f0f8ff); padding: .45rem; }
    .kp-eyebrow { color: #6b52eb; text-transform: uppercase; font-size: .72rem; letter-spacing: .16em; font-weight: 850; margin-bottom: .4rem; }
    .kp-subtitle { color: var(--kp-muted); font-size: 1rem; line-height: 1.7; max-width: 760px; margin-bottom: 1.2rem; }
    .kp-hero { background: linear-gradient(120deg,#6657f5 0%,#8d62ef 48%,#4388f5 100%); color: white; border-radius: 22px; padding: 1.45rem 1.6rem; margin: .4rem 0 1.2rem; box-shadow: 0 14px 34px rgba(102,87,245,.22); }
    .kp-hero-title { color: white; font-size: 1.45rem; font-weight: 850; letter-spacing: -.04em; margin-bottom: .35rem; }
    .kp-hero-copy { color: rgba(255,255,255,.88); font-size: .93rem; line-height: 1.6; }
    .kp-stat { border: 1px solid rgba(255,255,255,.8); border-radius: 18px; padding: 1.05rem 1.15rem; min-height: 112px; box-shadow: 0 8px 24px rgba(55,54,120,.07); }
    .kp-stat-purple { background: linear-gradient(135deg,#f0edff,#fff); }
    .kp-stat-blue { background: linear-gradient(135deg,#e7f2ff,#fff); }
    .kp-stat-teal { background: linear-gradient(135deg,#e2fbf5,#fff); }
    .kp-stat-orange { background: linear-gradient(135deg,#fff0e2,#fff); }
    .kp-stat-label { color: #656985; font-size: .78rem; font-weight: 750; margin-bottom: .45rem; }
    .kp-stat-value { color: var(--kp-ink); font-size: 1.75rem; font-weight: 850; letter-spacing: -.05em; }
    .kp-stat-note { color: #7c86a0; font-size: .72rem; margin-top: .2rem; }
    div.stButton > button[kind="primary"] { border-radius: 12px; font-weight: 750; border: 0; background: linear-gradient(100deg,#6657f5,#4388f5); box-shadow: 0 6px 16px rgba(98,91,246,.22); }
    div.stButton > button { border-radius: 11px; }
    div[data-testid="stExpander"] { border-radius: 14px; border-color: var(--kp-line); background: rgba(255,255,255,.9); }
    [data-testid="stRadio"] label { font-weight: 650; }
    .kp-section { background: rgba(255,255,255,.78); border: 1px solid #e9e7f8; border-radius: 18px; padding: 1.1rem 1.25rem; }
    @media (max-width: 700px) { .kp-stat { padding: .8rem; min-height: 90px; } .kp-stat-value { font-size: 1.35rem; } .kp-hero { padding: 1.1rem; } }
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
doc_total = len(documents) if health else 0
indexed_chunks = sum(int(d.get("chunk_count", 0)) for d in documents) if health else 0
status_label = "Connected" if health else "Offline"
status_color = "#0f9f78" if health else "#d97706"
page = st.sidebar.radio(
    "WORKSPACE",
    ["Overview", "Documents", "AI Assistant", "Analytics", "Settings"],
    index=0,
    key="kp_page",
    help="Choose a workspace area.",
)
st.sidebar.markdown("---")
st.markdown('<div class="kp-eyebrow">KNOWLEDGEPILOT AI · YOUR KNOWLEDGE WORKSPACE</div>', unsafe_allow_html=True)

if page == "Overview":
    st.title("Your knowledge, all in one place.")
    st.markdown('<div class="kp-subtitle">Bring your documents together, ask better questions, and trace every answer back to its source.</div>', unsafe_allow_html=True)
    st.markdown('<div class="kp-hero"><div class="kp-hero-title">✨ Welcome to your AI knowledge workspace</div><div class="kp-hero-copy">Upload PDFs or text files from the sidebar. KnowledgePilot indexes their content so you can ask questions and inspect the passages behind each answer.</div></div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    cards = [
        (m1, "📚 Indexed documents", f"{doc_total}", "Files in your workspace", "kp-stat-purple"),
        (m2, "🧩 Searchable passages", f"{indexed_chunks:,}", "Chunks ready for retrieval", "kp-stat-blue"),
        (m3, "● Backend status", status_label, "Live API health check", "kp-stat-teal" if health else "kp-stat-orange"),
    ]
    for col, label, value, note, color in cards:
        with col:
            st.markdown(f'<div class="kp-stat {color}"><div class="kp-stat-label">{label}</div><div class="kp-stat-value" style="font-size:{"1.35rem" if value in ("Connected","Offline") else "1.75rem"};color:{status_color if value in ("Connected","Offline") else "var(--kp-ink)"}">{value}</div><div class="kp-stat-note">{note}</div></div>', unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("📂 Recent documents")
        if documents:
            for doc in documents[:5]:
                with st.container(border=True):
                    st.markdown(f"**📄 {doc['filename']}**")
                    st.caption(f"{doc['page_count']} page(s) · {doc['chunk_count']} searchable passages")
        else:
            st.info("Your workspace is ready. Upload a PDF or TXT file from the sidebar to get started.")
    with right:
        st.subheader("💡 Try asking")
        st.caption("Select a prompt to open the assistant with a useful starting question.")
        for prompt in ["Summarize the key points in my documents", "What are the main recommendations?", "Compare the most important findings"]:
            if st.button(prompt, key=f"prompt_{prompt}", width="stretch", disabled=not documents):
                ss.messages.append({"role": "user", "content": prompt})
                ss.kp_page = "AI Assistant"
                st.rerun()
    st.caption("AI-generated answers may be incomplete or incorrect. Check source passages before relying on important information.")

elif page == "Documents":
    st.title("📁 Document library")
    st.markdown('<div class="kp-subtitle">Manage indexed files and see how much searchable content each document contributes.</div>', unsafe_allow_html=True)
    st.markdown('<div class="kp-hero"><div class="kp-hero-title">Your files, organized</div><div class="kp-hero-copy">Upload PDF or TXT files using the sidebar. Removing a file also removes its index entries.</div></div>', unsafe_allow_html=True)
    if documents:
        st.write(f"**{len(documents)} document(s) · {indexed_chunks:,} searchable passages**")
        for doc in documents:
            with st.container(border=True):
                a, b = st.columns([4, 1], vertical_alignment="center")
                a.markdown(f"**📄 {doc['filename']}**")
                a.caption(f"{doc['page_count']} page(s) · {doc['chunk_count']} chunks")
                with b.popover("Manage"):
                    st.write("Delete this document and its index entries?")
                    if st.button("Delete document", key=f"library_del_{doc['document_id']}", type="primary"):
                        delete_document(doc)
    else:
        st.info("No documents indexed yet. Upload a PDF or TXT file from the sidebar.")

elif page == "AI Assistant":
    st.title("💬 AI Assistant")
    st.markdown('<div class="kp-subtitle">Ask questions in plain language and inspect the sources used to answer.</div>', unsafe_allow_html=True)
    st.markdown('<div class="kp-hero"><div class="kp-hero-title">Ask your documents anything</div><div class="kp-hero-copy">Answers are grounded in retrieved passages. Open citations to review the original context.</div></div>', unsafe_allow_html=True)
    if not health:
        st.error(f"Backend not reachable at {API_URL}. Start the API and refresh.")
    elif not documents:
        st.info("Upload a document from the sidebar before starting a conversation.")
    for msg in ss.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                render_assistant(msg)
    if question := st.chat_input("Ask a question about your documents", disabled=health is None or not documents):
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

elif page == "Analytics":
    st.title("📊 Workspace analytics")
    st.markdown('<div class="kp-subtitle">A transparent snapshot of the content currently indexed by your backend.</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="kp-stat kp-stat-purple"><div class="kp-stat-label">📚 Documents</div><div class="kp-stat-value">{doc_total}</div><div class="kp-stat-note">Currently indexed</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="kp-stat kp-stat-blue"><div class="kp-stat-label">🧩 Searchable passages</div><div class="kp-stat-value">{indexed_chunks:,}</div><div class="kp-stat-note">Available to retrieval</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="kp-stat {"kp-stat-teal" if health else "kp-stat-orange"}"><div class="kp-stat-label">⚡ API status</div><div class="kp-stat-value" style="font-size:1.35rem;color:{status_color}">{status_label}</div><div class="kp-stat-note">Live health check</div></div>', unsafe_allow_html=True)
    st.subheader("Content breakdown")
    if documents:
        for doc in documents:
            st.write(f"**{doc['filename']}** · {doc['chunk_count']} chunks")
            st.progress(min(1.0, int(doc.get("chunk_count", 0)) / max(indexed_chunks, 1)))
    else:
        st.info("Upload documents to see your content breakdown.")
    st.caption("Analytics are based on live document metadata. No estimated accuracy or usage metrics are fabricated.")

elif page == "Settings":
    st.title("⚙️ Workspace settings")
    st.markdown('<div class="kp-subtitle">Review retrieval controls, model readiness, and privacy considerations.</div>', unsafe_allow_html=True)
    st.subheader("🔎 Retrieval")
    st.write(f"**Passages per question (top-k):** {top_k}")
    st.write(f"**Minimum relevance score:** {min_score if min_score is not None else 'Server default'}")
    st.write(f"**Document scope:** {len(selected)} selected" if selected else "**Document scope:** All indexed documents")
    st.subheader("🧠 AI services")
    if health:
        llm = health["llm"]
        st.write(f"**Answer provider:** {health['llm_provider']}")
        st.write(f"**Answer model status:** {'Ready' if llm['ok'] else 'Needs configuration'}")
        st.caption(llm["detail"])
        st.write(f"**Embedding service:** {health['embedding']['detail']}")
        if not llm["ok"]:
            st.warning("The answer model is not ready. See the README for configuration steps.")
        if health.get("llm_is_external"):
            st.warning("Privacy note: questions and retrieved excerpts are sent to an external LLM service.")
        else:
            st.success("The health endpoint does not report an external LLM service.")
    else:
        st.error(f"Backend offline at {API_URL}. Check the API service and refresh.")
    st.subheader("🛡️ Responsible use")
    st.info("Answers can be wrong or incomplete. Open citations and verify original document text before making important decisions.")
    if st.button("Clear conversation history", type="primary"):
        ss.messages = []
        st.rerun()
