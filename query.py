from dotenv import load_dotenv
import os

from qdrant_client import QdrantClient

from langchain_mistralai import MistralAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate

from huggingface_hub import InferenceClient


load_dotenv()


# ============================================================
# Environment Variables
# ============================================================

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")


# ============================================================
# Validate environment variables
# ============================================================

if not QDRANT_URL:
    raise ValueError("QDRANT_URL is missing from .env")

if not QDRANT_API_KEY:
    raise ValueError("QDRANT_API_KEY is missing from .env")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is missing from .env")


# ============================================================
# Step 1: Embedding Model
# ============================================================

# IMPORTANT:
# Your existing Qdrant documents were embedded using
# MistralAIEmbeddings.
#
# Therefore, we MUST use the same embedding model
# when converting the user's query into a vector.

embedding_model = MistralAIEmbeddings()


# ============================================================
# Step 2: Connect to Qdrant Cloud
# ============================================================

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)


vectorstore = QdrantVectorStore(
    client=client,
    collection_name="docs_collection",
    embedding=embedding_model
)


# ============================================================
# Step 3: Create Retriever
# ============================================================

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)


# ============================================================
# Step 4: Hugging Face DeepSeek
# ============================================================

hf_client = InferenceClient(
    api_key=HF_TOKEN,
    provider="auto"
)


MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"


# ============================================================
# Step 5: Prompt Template
# ============================================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful Documentation Copilot.

You must answer the developer's question using ONLY
the provided documentation context.

Rules:

1. Do NOT use outside knowledge.
2. Do NOT invent or hallucinate information.
3. If the answer is not present in the context,
   say exactly:

"I could not find the answer in the document."

4. Give a clear and technically accurate answer.
5. When code is relevant, provide a code example.
"""
        ),

        (
            "human",
            """
Context:

{context}

Question:

{question}
"""
        )
    ]
)


# ============================================================
# Step 6: Start RAG System
# ============================================================

print("🤖 Documentation Copilot ready")
print("--------------------------------")
print("Embedding : Mistral")
print("Vector DB : Qdrant")
print("LLM       : DeepSeek-R1 via Hugging Face")
print("--------------------------------")
print("Press 0 to exit")


# ============================================================
# Step 7: Interactive RAG Loop
# ============================================================

while True:

    query = input("\nDeveloper: ")

    if query.strip() == "0":
        print("👋 Exiting...")
        break


    # ========================================================
    # Retrieve relevant documents
    # ========================================================

    print("\n🔎 Searching documentation...")

    docs = retriever.invoke(query)


    # ========================================================
    # Check retrieval
    # ========================================================

    if not docs:
        print("\n🤖 Copilot:")
        print("I could not find the answer in the document.")
        continue


    # ========================================================
    # Display retrieved documents
    # ========================================================

    print(f"\n📚 Retrieved {len(docs)} documents:")

    for i, doc in enumerate(docs, start=1):

        print(f"\n--- Document {i} ---")

        print(
            "Source:",
            doc.metadata.get("source", "Unknown")
        )

        print(
            "Version:",
            doc.metadata.get("version", "Unknown")
        )

        print(
            "Content:",
            doc.page_content
        )


    # ========================================================
    # Build context
    # ========================================================

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )


    # ========================================================
    # Create LangChain prompt messages
    # ========================================================

    messages = prompt.format_messages(
        context=context,
        question=query
    )


    # ========================================================
    # Convert LangChain messages to Hugging Face format
    # ========================================================

    hf_messages = []

    for message in messages:

        if message.type == "system":
            role = "system"

        elif message.type == "human":
            role = "user"

        elif message.type == "ai":
            role = "assistant"

        else:
            role = "user"

        hf_messages.append(
            {
                "role": role,
                "content": message.content
            }
        )


    # ========================================================
    # Call DeepSeek
    # ========================================================

    print("\n🧠 DeepSeek is generating answer...")

    response = hf_client.chat.completions.create(
        model=MODEL_NAME,
        messages=hf_messages,
        max_tokens=1000
    )


    # ========================================================
    # Extract response
    # ========================================================

    answer = response.choices[0].message.content


    # ========================================================
    # Display final answer
    # ========================================================

    print("\n🤖 Copilot:")
    print(answer)