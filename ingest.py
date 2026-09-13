import os

from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

from langchain_community.document_loaders import WebBaseLoader


# -----------------------------
# 1. Load Environment Variables
# -----------------------------

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")


# -----------------------------
# 2. Scrape Documentation
# -----------------------------

url = "https://docs.langchain.com/oss/python/langchain/overview"

loader = WebBaseLoader(url)

docs = loader.load()

print(f"Loaded {len(docs)} document(s)")

raw_text = " ".join(
    doc.page_content
    for doc in docs
)

print(f"Raw text length: {len(raw_text)}")


# -----------------------------
# 3. Normalize Text
# -----------------------------

clean_text = raw_text.replace("\n", " ").strip()


# -----------------------------
# 4. Embedding Model
# -----------------------------

embedding_model = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-large-en-v1.5",
    huggingfacehub_api_token=HF_TOKEN
)


# -----------------------------
# 5. Recursive Chunking
# -----------------------------

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)

recursive_chunks = recursive_splitter.split_text(
    clean_text
)
print(f"Recursive chunks: {len(recursive_chunks)}")


# -----------------------------
# 6. Semantic Chunking
# -----------------------------

semantic_splitter = SemanticChunker(
    embeddings=embedding_model
)

semantic_chunks = []

for chunk in recursive_chunks:
    chunks = semantic_splitter.split_text(chunk)
    semantic_chunks.extend(chunks)


print(f"Semantic chunks: {len(semantic_chunks)}")


# -----------------------------
# 7. Create Dense Embeddings
# -----------------------------

vectors = embedding_model.embed_documents(
    semantic_chunks
)
vector_size = len(vectors[0])
print(f"Vector size: {vector_size}")


# -----------------------------
# 8. Connect to Qdrant
# -----------------------------

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

collection_name = "langchain_docs"


# -----------------------------
# 9. Create Collection
# -----------------------------

if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )
    print(f"Created collection: {collection_name}")


# -----------------------------
# 10. Upload Vectors
# -----------------------------

points = [
    PointStruct(
        id=i,
        vector=vectors[i],
        payload={
            "source": f"chunk_{i}",
            "text": semantic_chunks[i]
        }
    )
    for i in range(len(semantic_chunks))
]


client.upsert(
    collection_name=collection_name,
    points=points
)


print("✅ LangChain docs ingested into Qdrant")