
# requirements:
# pip install chromadb ollama langchain langchain-community

import ollama
import chromadb
from chromadb.utils import embedding_functions

# ── 1. Setup ChromaDB ──────────────────────────────────────────────────────────

client = chromadb.PersistentClient(path="./chroma_db")

# Use Ollama embeddings (nomic-embed-text is fast & good)
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text",
)

collection = client.get_or_create_collection(
    name="knowledge_base",
    embedding_function=ollama_ef,
    metadata={"hnsw:space": "cosine"},
)

# ── 2. Populate ChromaDB with documents ───────────────────────────────────────

documents = [
    "Python is a high-level, interpreted programming language known for its simplicity.",
    "ChromaDB is an open-source vector database designed for AI applications.",
    "Ollama allows you to run large language models locally on your machine.",
    "RAG (Retrieval-Augmented Generation) combines retrieval systems with LLMs.",
    "Vector embeddings are numerical representations of text used for semantic search.",
    "LangChain is a framework for building applications powered by language models.",
    "FastAPI is a modern web framework for building APIs with Python.",
    "Docker is a platform for developing and running applications in containers.",
]

# Add documents (skip if already added)
existing = collection.count()
if existing == 0:
    collection.add(
        documents=documents,
        ids=[f"doc_{i}" for i in range(len(documents))],
        metadatas=[{"source": "manual", "index": i} for i in range(len(documents))],
    )
    print(f"✅ Added {len(documents)} documents to ChromaDB")
else:
    print(f"ℹ️  Collection already has {existing} documents")


# ── 3. RAG Query Function ──────────────────────────────────────────────────────

def rag_query(question: str, n_results: int = 3, model: str = "llama3.2") -> str:
    # Step 1: Retrieve relevant documents
    results = collection.query(
        query_texts=[question],
        n_results=n_results,
    )

    retrieved_docs = results["documents"][0]
    
    print(f"\n📚 Retrieved {len(retrieved_docs)} relevant chunks:")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"  {i}. {doc[:80]}...")

    # Step 2: Build prompt with context
    context = "\n".join(f"- {doc}" for doc in retrieved_docs)
    
    prompt = f"""You are a helpful assistant. Use the following context to answer the question.
If the answer isn't in the context, say so clearly.

Context:
{context}

Question: {question}

Answer:"""

    # Step 3: Generate response with Ollama
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    return response["message"]["content"]


# ── 4. Add documents dynamically ──────────────────────────────────────────────

def add_document(text: str, doc_id: str = None, metadata: dict = None):
    count = collection.count()
    doc_id = doc_id or f"doc_{count}"
    metadata = metadata or {"source": "dynamic"}
    
    collection.add(
        documents=[text],
        ids=[doc_id],
        metadatas=[metadata],
    )
    print(f"✅ Added document: '{text[:50]}...' with id '{doc_id}'")


# ── 5. Run example queries ─────────────────────────────────────────────────────

if __name__ == "__main__":
    questions = [
        "What is ChromaDB used for?",
        "How does RAG work?",
        "Can I run LLMs locally?",
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"❓ Question: {question}")
        answer = rag_query(question)
        print(f"\n🤖 Answer:\n{answer}")

    # Example: add a new document
    # add_document("Kubernetes is a container orchestration platform.", doc_id="doc_k8s")