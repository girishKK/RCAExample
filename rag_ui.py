# rag_ui.py
# ── Install dependencies ───────────────────────────────────────────────────────
# pip install streamlit chromadb ollama pypdf python-docx langchain-text-splitters
#             requests beautifulsoup4

import io
import hashlib
import requests
import streamlit as st
import ollama
import chromadb
from chromadb.utils import embedding_functions
from bs4 import BeautifulSoup

# Optional imports — gracefully degrade if not installed
try:
    import pypdf
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from docx import Document as DocxDocument
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    SPLITTER_AVAILABLE = True
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        SPLITTER_AVAILABLE = True
    except ImportError:
        SPLITTER_AVAILABLE = False

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .source-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #6366f1;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.83rem;
        color: #334155;
        line-height: 1.6;
    }
    .source-meta {
        font-size: 0.72rem;
        color: #6366f1;
        margin-bottom: 6px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .sim-badge {
        display: inline-block;
        background: #ede9fe;
        color: #6366f1;
        border: 1px solid #c4b5fd;
        border-radius: 4px;
        padding: 1px 7px;
        font-size: 0.7rem;
        margin-left: 8px;
    }
    .file-tag {
        display: inline-block;
        background: #dcfce7;
        color: #166534;
        border: 1px solid #86efac;
        border-radius: 3px;
        padding: 1px 8px;
        font-size: 0.72rem;
        margin: 2px 3px;
    }
    .status-bar {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 8px 14px;
        font-size: 0.82rem;
        color: #64748b;
        margin-bottom: 12px;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 60) -> list[str]:
    if SPLITTER_AVAILABLE:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return splitter.split_text(text)
    # Naive fallback splitter (by words)
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i: i + chunk_size]))
        i += chunk_size - chunk_overlap
    return chunks


def extract_text_from_pdf(file_bytes: bytes) -> str:
    if not PDF_SUPPORT:
        st.error("pypdf not installed. Run: pip install pypdf")
        return ""
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text_from_docx(file_bytes: bytes) -> str:
    if not DOCX_SUPPORT:
        st.error("python-docx not installed. Run: pip install python-docx")
        return ""
    doc = DocxDocument(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract_text_from_url(url: str) -> tuple[str, str]:
    """Returns (text, title_or_error)"""
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        title = soup.title.string.strip() if soup.title else url
        text = soup.get_text(separator=" ", strip=True)
        return text, title
    except Exception as e:
        return "", str(e)


def file_hash(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()[:8]


def add_chunks_to_collection(chunks: list[str], source_name: str, collection) -> int:
    """Upsert chunks into ChromaDB. Returns number of chunks added."""
    if not chunks:
        return 0
    ids = [f"{source_name}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source_name}] * len(chunks)
    collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
    return len(chunks)


# ── Init ChromaDB (cached across reruns) ──────────────────────────────────────
@st.cache_resource
def init_chromadb():
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text",
    )
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="knowledge_base",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


@st.cache_data(ttl=5)
def get_available_models():
    try:
        models = ollama.list()
        return [m["name"] for m in models["models"]]
    except Exception:
        return ["llama3.2", "mistral", "phi3"]


# ── Session State ──────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ingested_sources" not in st.session_state:
    st.session_state.ingested_sources = []

collection = init_chromadb()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ RAG Assistant")
    st.divider()

    # Model & retrieval settings
    st.markdown("**Model**")
    available_models = get_available_models()
    selected_model = st.selectbox("Ollama model", available_models, label_visibility="collapsed")

    col1, col2 = st.columns(2)
    with col1:
        n_results = st.number_input("Chunks", min_value=1, max_value=10, value=3,
                                    help="Number of chunks retrieved per query")
    with col2:
        chunk_size = st.number_input("Chunk size", min_value=100, max_value=2000, value=500, step=100,
                                     help="Characters per chunk when ingesting")

    show_sources = st.toggle("Show sources", value=True)
    stream_response = st.toggle("Stream response", value=True)

    st.divider()

    # Knowledge Base stats
    doc_count = collection.count()
    st.metric("📚 Chunks in knowledge base", doc_count)

    if st.session_state.ingested_sources:
        with st.expander("Ingested sources"):
            for src in st.session_state.ingested_sources:
                st.markdown(f'<span class="file-tag">{src}</span>', unsafe_allow_html=True)

    st.divider()

    # ── Document ingestion tabs ────────────────────────────────────────────────
    st.markdown("**➕ Add to Knowledge Base**")
    tab_file, tab_url, tab_text = st.tabs(["📁 Files", "🌐 URL", "✏️ Text"])

    # Tab 1 — File Upload
    with tab_file:
        accepted_types = []
        if PDF_SUPPORT:
            accepted_types.append("pdf")
        if DOCX_SUPPORT:
            accepted_types.append("docx")
        accepted_types += ["txt", "md"]

        uploaded_files = st.file_uploader(
            "Upload files",
            type=accepted_types,
            accept_multiple_files=True,
            label_visibility="collapsed",
            help=f"Supported: {', '.join(t.upper() for t in accepted_types)}",
        )
        if uploaded_files:
            st.caption(f"{len(uploaded_files)} file(s) selected")

        if st.button("⬆️ Ingest Files", use_container_width=True, type="primary",
                     disabled=not uploaded_files):
            total_chunks = 0
            progress = st.progress(0)
            for idx, f in enumerate(uploaded_files):
                with st.spinner(f"Processing {f.name}…"):
                    raw = f.read()
                    ext = f.name.rsplit(".", 1)[-1].lower()

                    if ext == "pdf":
                        text = extract_text_from_pdf(raw)
                    elif ext == "docx":
                        text = extract_text_from_docx(raw)
                    else:
                        text = raw.decode("utf-8", errors="ignore")

                    if text.strip():
                        source_id = f"{f.name}_{file_hash(raw)}"
                        chunks = chunk_text(text, chunk_size=chunk_size)
                        n = add_chunks_to_collection(chunks, source_id, collection)
                        total_chunks += n
                        display_name = f.name
                        if display_name not in st.session_state.ingested_sources:
                            st.session_state.ingested_sources.append(display_name)
                    else:
                        st.warning(f"No text extracted from {f.name}")

                progress.progress((idx + 1) / len(uploaded_files))

            progress.empty()
            st.success(f"✅ Added {total_chunks} chunks from {len(uploaded_files)} file(s)")
            st.rerun()

    # Tab 2 — URL
    with tab_url:
        url_input = st.text_input("Web URL", placeholder="https://example.com/article",
                                   label_visibility="collapsed")

        if st.button("⬇️ Fetch & Ingest", use_container_width=True, type="primary",
                     disabled=not url_input):
            with st.spinner("Fetching page…"):
                text, title = extract_text_from_url(url_input)
            if text:
                chunks = chunk_text(text, chunk_size=chunk_size)
                source_id = f"url_{file_hash(url_input.encode())}"
                n = add_chunks_to_collection(chunks, source_id, collection)
                display_name = title[:40]
                if display_name not in st.session_state.ingested_sources:
                    st.session_state.ingested_sources.append(display_name)
                st.success(f"✅ Added {n} chunks from:\n{title[:60]}")
                st.rerun()
            else:
                st.error(f"Failed to fetch URL:\n{title}")

    # Tab 3 — Paste Text
    with tab_text:
        paste_text = st.text_area("Paste text", height=140,
                                   placeholder="Paste any text here…",
                                   label_visibility="collapsed")
        paste_label = st.text_input("Label / source name",
                                     placeholder="e.g. my-notes")

        if st.button("➕ Add Text", use_container_width=True, type="primary",
                     disabled=not paste_text.strip()):
            label = paste_label.strip() or f"paste_{file_hash(paste_text.encode())}"
            chunks = chunk_text(paste_text, chunk_size=chunk_size)
            n = add_chunks_to_collection(chunks, label, collection)
            if label not in st.session_state.ingested_sources:
                st.session_state.ingested_sources.append(label)
            st.success(f"✅ Added {n} chunks as '{label}'")
            st.rerun()

    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("💣 Delete ALL Documents", use_container_width=True):
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)
        st.session_state.ingested_sources = []
        st.warning("All documents deleted.")
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ══════════════════════════════════════════════════════════════════════════════
st.title("🧠 RAG Assistant")
st.markdown(
    f'<div class="status-bar">'
    f'model: <strong>{selected_model}</strong> &nbsp;|&nbsp; '
    f'chunks in db: <strong>{collection.count()}</strong> &nbsp;|&nbsp; '
    f'retrieve top-<strong>{n_results}</strong>'
    f'</div>',
    unsafe_allow_html=True,
)

# Render chat history
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        if show_sources and entry.get("sources"):
            with st.expander("📎 Retrieved chunks"):
                for i, (doc, dist, meta) in enumerate(
                    zip(entry["sources"], entry["distances"], entry["metadatas"]), 1
                ):
                    sim = round((1 - dist) * 100, 1)
                    src = meta.get("source", "unknown")
                    st.markdown(
                        f'<div class="source-card">'
                        f'<div class="source-meta">chunk {i} — {src}'
                        f'<span class="sim-badge">{sim}% match</span></div>'
                        f'{doc}</div>',
                        unsafe_allow_html=True,
                    )

# Empty state hint
if collection.count() == 0 and not st.session_state.chat_history:
    st.info(
        "👈 Your knowledge base is empty.\n\n"
        "Use the sidebar to upload **PDF / DOCX / TXT / MD** files, "
        "fetch a **URL**, or paste text directly.",
        icon="📭",
    )

# Chat input
query = st.chat_input("Ask something from your knowledge base…")

if query:
    if collection.count() == 0:
        st.warning("⚠️ Knowledge base is empty — add documents first via the sidebar.")
        st.stop()

    with st.chat_message("user"):
        st.write(query)

    # Retrieve relevant chunks
    with st.spinner("🔍 Searching knowledge base…"):
        results = collection.query(query_texts=[query], n_results=n_results)
        retrieved_docs = results["documents"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]

    # Build RAG prompt
    context = "\n".join(
        f"[{m.get('source', '?')}]: {doc}"
        for doc, m in zip(retrieved_docs, metadatas)
    )
    prompt = f"""You are a helpful assistant. Answer the question using only the context below.
If the answer isn't in the context, say so clearly. Cite which source(s) you used.

Context:
{context}

Question: {query}

Answer:"""

    # Generate response
    with st.chat_message("assistant"):
        full_response = ""

        if stream_response:
            placeholder = st.empty()
            try:
                stream = ollama.chat(
                    model=selected_model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                )
                for chunk in stream:
                    full_response += chunk["message"]["content"]
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"Ollama error: {e}\n\nMake sure Ollama is running: `ollama serve`")
                st.stop()
        else:
            try:
                resp = ollama.chat(
                    model=selected_model,
                    messages=[{"role": "user", "content": prompt}],
                )
                full_response = resp["message"]["content"]
                st.markdown(full_response)
            except Exception as e:
                st.error(f"Ollama error: {e}")
                st.stop()

        # Show source chunks
        if show_sources:
            with st.expander("📎 Retrieved chunks"):
                for i, (doc, dist, meta) in enumerate(
                    zip(retrieved_docs, distances, metadatas), 1
                ):
                    sim = round((1 - dist) * 100, 1)
                    src = meta.get("source", "unknown")
                    st.markdown(
                        f'<div class="source-card">'
                        f'<div class="source-meta">chunk {i} — {src}'
                        f'<span class="sim-badge">{sim}% match</span></div>'
                        f'{doc}</div>',
                        unsafe_allow_html=True,
                    )

    # Persist to chat history
    st.session_state.chat_history.append({
        "question": query,
        "answer": full_response,
        "sources": retrieved_docs,
        "distances": distances,
        "metadatas": metadatas,
    })
