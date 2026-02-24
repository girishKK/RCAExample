# rag_ui.py
# pip install streamlit chromadb ollama

import streamlit as st
import ollama
import chromadb
from chromadb.utils import embedding_functions

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🧠",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .source-box {
        background: #1e1e2e;
        border-left: 3px solid #7c3aed;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.85rem;
        color: #cdd6f4;
    }
    .answer-box {
        background: #1a2744;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 10px;
        color: #e2e8f0;
    }
    .metric-card {
        background: #2a2a3e;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
    }
    .stTextArea textarea { font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

# ── ChromaDB & Ollama Setup (cached) ───────────────────────────────────────────
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

@st.cache_data
def get_available_models():
    try:
        models = ollama.list()
        return [m["name"] for m in models["models"]]
    except Exception:
        return ["llama3.2", "mistral", "phi3"]

# ── Session State ──────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "collection" not in st.session_state:
    st.session_state.collection = init_chromadb()

collection = st.session_state.collection

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    st.divider()

    # Model selection
    available_models = get_available_models()
    selected_model = st.selectbox("🤖 Ollama Model", available_models, index=0)

    # RAG settings
    n_results = st.slider("📄 Retrieved Chunks", min_value=1, max_value=8, value=3)
    show_sources = st.toggle("Show Source Chunks", value=True)
    stream_response = st.toggle("Stream Response", value=True)

    st.divider()

    # Document count
    doc_count = collection.count()
    st.metric("📚 Documents in DB", doc_count)

    st.divider()

    # Add documents section
    st.subheader("➕ Add Documents")
    new_doc = st.text_area("Paste text to add:", height=120, placeholder="Enter document text here...")
    doc_id = st.text_input("Custom ID (optional)", placeholder="e.g. doc_001")
    doc_source = st.text_input("Source label (optional)", placeholder="e.g. wikipedia")

    if st.button("Add to Knowledge Base", use_container_width=True, type="primary"):
        if new_doc.strip():
            try:
                count = collection.count()
                _id = doc_id.strip() if doc_id.strip() else f"doc_{count}"
                _meta = {"source": doc_source.strip() if doc_source.strip() else "manual"}
                collection.add(documents=[new_doc], ids=[_id], metadatas=[_meta])
                st.success(f"✅ Added as `{_id}`")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter some text first.")

    st.divider()

    # Clear chat history
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ── Main Area ──────────────────────────────────────────────────────────────────
st.title("🧠 RAG Assistant")
st.caption(f"Powered by **Ollama** · **ChromaDB** · Model: `{selected_model}`")
st.divider()

# ── Chat History Display ───────────────────────────────────────────────────────
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        if show_sources and entry.get("sources"):
            with st.expander("📎 Source Chunks Used"):
                for i, src in enumerate(entry["sources"], 1):
                    st.markdown(
                        f'<div class="source-box"><strong>Chunk {i}:</strong> {src}</div>',
                        unsafe_allow_html=True,
                    )

# ── Query Input ────────────────────────────────────────────────────────────────
query = st.chat_input("Ask something from your knowledge base...")

if query:
    if collection.count() == 0:
        st.warning("⚠️ Your knowledge base is empty. Add some documents in the sidebar first.")
        st.stop()

    # Show user message immediately
    with st.chat_message("user"):
        st.write(query)

    # Retrieve relevant chunks
    with st.spinner("🔍 Searching knowledge base..."):
        results = collection.query(query_texts=[query], n_results=n_results)
        retrieved_docs = results["documents"][0]
        distances = results["distances"][0]

    # Build prompt
    context = "\n".join(f"- {doc}" for doc in retrieved_docs)
    prompt = f"""You are a helpful assistant. Use the following context to answer the question.
If the answer isn't covered by the context, say so clearly.

Context:
{context}

Question: {query}

Answer:"""

    # Generate and display response
    with st.chat_message("assistant"):
        if stream_response:
            response_placeholder = st.empty()
            full_response = ""
            try:
                stream = ollama.chat(
                    model=selected_model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                )
                for chunk in stream:
                    token = chunk["message"]["content"]
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"Ollama error: {e}. Is Ollama running? (`ollama serve`)")
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

        # Show sources
        if show_sources:
            with st.expander("📎 Source Chunks Used"):
                for i, (doc, dist) in enumerate(zip(retrieved_docs, distances), 1):
                    similarity = round((1 - dist) * 100, 1)
                    st.markdown(
                        f'<div class="source-box"><strong>Chunk {i}</strong> '
                        f'<span style="color:#a78bfa">({similarity}% match)</span><br>{doc}</div>',
                        unsafe_allow_html=True,
                    )

    # Save to chat history
    st.session_state.chat_history.append({
        "question": query,
        "answer": full_response,
        "sources": retrieved_docs,
    })