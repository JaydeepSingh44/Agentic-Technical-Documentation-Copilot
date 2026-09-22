import os
import requests

from dotenv import load_dotenv

from Web_Search_Agent import find_documentation_url

from ingest import (
    ingest_url,
    embedding_model,
    client,
    COLLECTION_NAME,
)

from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is missing from .env")


# ============================================================
# DEEPSEEK LLM
# ============================================================

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=2000,
    temperature=0.1,
)

model = ChatHuggingFace(
    llm=llm
)


# ============================================================
# URL VALIDATION
# ============================================================

def validate_url(url: str) -> bool:

    try:

        response = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "Chrome/140.0 Safari/537.36"
                )
            },
        )

        print(
            f"URL validation status: "
            f"{response.status_code}"
        )

        return response.status_code == 200

    except Exception as e:

        print(
            f"URL validation failed: {e}"
        )

        return False


# ============================================================
# QDRANT RETRIEVAL
# ============================================================

def retrieve_chunks(
    question: str,
    limit: int = 5,
):

    print("\n" + "=" * 60)
    print("RETRIEVING FROM QDRANT")
    print("=" * 60)

    print(
        f"Embedding question: {question}"
    )

    # Convert question into embedding
    query_vector = embedding_model.embed_query(
        question
    )

    # Search Qdrant
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )

    points = results.points

    print(
        f"\nRetrieved {len(points)} chunks"
    )

    # Display retrieved chunks
    for i, point in enumerate(
        points,
        start=1,
    ):

        payload = point.payload

        print("\n" + "-" * 60)

        print(
            f"RESULT {i}"
        )

        print(
            f"Score: {point.score}"
        )

        print(
            f"URL: {payload.get('url')}"
        )

        print(
            f"Heading: "
            f"{payload.get('heading')}"
        )

        print("\nText:")

        print(
            payload.get(
                "text",
                "",
            )[:1000]
        )

    return points


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(results):

    context_parts = []

    for i, point in enumerate(
        results,
        start=1,
    ):

        payload = point.payload

        text = payload.get(
            "text",
            "",
        )

        url = payload.get(
            "url",
            "",
        )

        heading = payload.get(
            "heading",
            "",
        )

        context_parts.append(
            f"""
============================================================
SOURCE {i}
============================================================

URL:
{url}

HEADING:
{heading}

CONTENT:
{text}
"""
        )

    return "\n".join(
        context_parts
    )


# ============================================================
# GENERATE FINAL ANSWER
# ============================================================

def generate_answer(
    question: str,
    results,
):

    print("\n" + "=" * 60)
    print("GENERATING FINAL ANSWER")
    print("=" * 60)

    context = build_context(
        results
    )

    prompt = f"""
You are a technical documentation assistant.

Your job is to answer the user's technical question
using ONLY the documentation provided in the context.

============================================================
USER QUESTION
============================================================

{question}

============================================================
DOCUMENTATION CONTEXT
============================================================

{context}

============================================================
RULES
============================================================

1. Answer the user's actual question directly.

2. Use ONLY information supported by the documentation.

3. Do NOT invent:
   - APIs
   - functions
   - classes
   - parameters
   - arguments
   - configuration options
   - methods
   - behavior

4. If the documentation contains code examples,
   use them as the basis for your answer.

5. If you provide code, make sure the code is
   supported by the documentation.

6. Explain the important parts of the code
   when necessary.

7. Cite the documentation using:
   [Source 1]
   [Source 2]
   etc.

8. The source number must correspond to the
   SOURCE number provided in the context.

9. If the documentation does not contain enough
   information to answer the question, clearly say:

   "The provided documentation does not contain
   enough information to answer this question."

10. Do not mention:
    - Qdrant
    - embeddings
    - retrieval
    - prompts
    - internal pipeline
    - web search agent

11. Do not discuss how the answer was generated.

12. Give a practical developer-focused answer.

13. If appropriate, structure the answer with:
    - explanation
    - code example
    - important notes

Return ONLY the final technical answer.
"""

    response = model.invoke(
        [
            {
                "role": "user",
                "content": prompt,
            }
        ]
    )

    return response.content


# ============================================================
# PRINT SOURCES
# ============================================================

def print_sources(results):

    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)

    seen_urls = set()

    source_number = 1

    for point in results:

        payload = point.payload

        url = payload.get(
            "url"
        )

        if not url:
            continue

        if url in seen_urls:
            continue

        seen_urls.add(url)

        print(
            f"[Source {source_number}] "
            f"{url}"
        )

        heading = payload.get(
            "heading"
        )

        if heading:
            print(
                f"Heading: {heading}"
            )

        print()

        source_number += 1


# ============================================================
# END-TO-END PIPELINE
# ============================================================

def run_pipeline(question: str):

    print("\n")

    print("=" * 70)
    print("TECHNICAL DOCUMENTATION COPILOT")
    print("=" * 70)


    # ========================================================
    # STEP 1
    # FIND DOCUMENTATION URL
    # ========================================================

    print("\n" + "=" * 60)
    print("STEP 1: SOURCE DISCOVERY")
    print("=" * 60)

    url = find_documentation_url(
        question
    )

    if not url:

        print(
            "\nCould not find "
            "documentation URL."
        )

        return


    print(
        f"\nDocumentation URL:\n{url}"
    )


    # ========================================================
    # STEP 2
    # VALIDATE URL
    # ========================================================

    print("\n" + "=" * 60)
    print("STEP 2: VALIDATING URL")
    print("=" * 60)

    if not validate_url(url):

        print(
            "\nURL is not reachable."
        )

        return


    # ========================================================
    # STEP 3
    # INGEST DOCUMENTATION
    # ========================================================

    print("\n" + "=" * 60)
    print("STEP 3: INGESTING DOCUMENTATION")
    print("=" * 60)

    ingestion_result = ingest_url(
        url
    )

    print(
        "\nIngestion result:"
    )

    print(
        ingestion_result
    )


    # ========================================================
    # STEP 4
    # RETRIEVE FROM QDRANT
    # ========================================================

    print("\n" + "=" * 60)
    print("STEP 4: RETRIEVING DOCUMENTATION")
    print("=" * 60)

    results = retrieve_chunks(
        question,
        limit=5,
    )

    if not results:

        print(
            "\nNo relevant documentation "
            "chunks were retrieved."
        )

        return


    # ========================================================
    # STEP 5
    # GENERATE FINAL ANSWER
    # ========================================================

    print("\n" + "=" * 60)
    print("STEP 5: GENERATING FINAL ANSWER")
    print("=" * 60)

    answer = generate_answer(
        question,
        results,
    )


    # ========================================================
    # STEP 6
    # DISPLAY ANSWER
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print()

    print(answer)


    # ========================================================
    # STEP 7
    # DISPLAY SOURCES
    # ========================================================

    print_sources(
        results
    )


    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 70)
    print("END-TO-END TEST COMPLETE")
    print("=" * 70)

    print(
        f"Source URL: {url}"
    )

    print(
        f"Retrieved chunks: "
        f"{len(results)}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    question = input(
        "\nAsk a developer question: "
    )

    run_pipeline(
        question
    )