import chromadb
import openai
import uuid

# Initialize OpenAI client
client = openai.OpenAI(api_key="sk-WJDrCfHP7jdl2eJdZVW07g")

# Create persistent Chroma client
chroma_client = chromadb.Client()

# Create collection
collection = chroma_client.create_collection(name="my_knowledge_base")

# Example documents
documents = [
    "Python is a high-level programming language.",
    "Vector databases store embeddings for similarity search.",
    "Large Language Models can use retrieval augmented generation.",
]

# Generate embeddings
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Add documents to vector DB
for doc in documents:
    embedding = get_embedding(doc)
    collection.add(
        ids=[str(uuid.uuid4())],
        documents=[doc],
        embeddings=[embedding]
    )

print("Vector DB populated successfully!")