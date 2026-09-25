"""
SourceGuard FastAPI RAG query module.

Important design:
- Retrieval works independently of the LLM.
- The Hugging Face LLM is created lazily, so retrieval/evaluation
  does NOT consume Hugging Face inference credits.
- `run_retrieval()` is used by evaluate.py for the 20-question test.
- `run_query()` is the full production path: retrieval -> threshold
  -> evidence check -> grounded answer -> citations.
"""

import os
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

from ingest2 import (
    embedding_model,
    client,
    COLLECTION_NAME,
)

load_dotenv()

TOP_K = 5

# Initial value. evaluate.py performs a threshold sweep.
RETRIEVAL_THRESHOLD = 0.55

CONTEXT_K = 5

_model = None


def get_model():
    """Create the Hugging Face model only when an LLM answer is requested."""
    global _model

    if _model is not None:
        return _model

    from langchain_huggingface import (
        ChatHuggingFace,
        HuggingFaceEndpoint,
    )

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN is missing from .env")

    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
        task="text-generation",
        huggingfacehub_api_token=hf_token,
        max_new_tokens=1200,
        temperature=0.0,
    )

    _model = ChatHuggingFace(llm=llm)
    return _model


def retrieve_chunks(question: str, limit: int = TOP_K):
    """Retrieve passages from the already-ingested Qdrant corpus only."""
    query_vector = embedding_model.embed_query(question)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )

    return results.points


def get_top_score(results) -> Optional[float]:
    if not results:
        return None
    return float(results[0].score)


def check_retrieval_strength(results, threshold: float = RETRIEVAL_THRESHOLD) -> bool:
    score = get_top_score(results)
    return score is not None and score >= threshold


def get_sources(results) -> List[Dict[str, Any]]:
    """Return passage-level source information."""
    sources = []

    for index, point in enumerate(results[:CONTEXT_K], start=1):
        payload = point.payload or {}

        sources.append(
            {
                "source_number": index,
                "url": payload.get("url", ""),
                "heading": payload.get("heading", ""),
                "chunk_id": payload.get("chunk_id", ""),
                "score": float(point.score),
                "passage": payload.get("text", ""),
            }
        )

    return sources


def build_context(results) -> str:
    parts = []

    for source in get_sources(results):
        parts.append(
            f"""
============================================================
SOURCE {source['source_number']}
============================================================

SOURCE URL:
{source['url']}

SECTION:
{source['heading']}

CHUNK ID:
{source['chunk_id']}

RETRIEVAL SCORE:
{source['score']:.4f}

PASSAGE:
{source['passage']}
"""
        )

    return "\n".join(parts)


def check_answerability(question: str, results) -> bool:
    """Use the LLM as a second evidence gate, only when explicitly requested."""
    model = get_model()
    context = build_context(results)

    prompt = f"""
You are an evidence verification system.

Determine whether the retrieved passages contain enough explicit
information to answer the user's question.

USER QUESTION:
{question}

RETRIEVED PASSAGES:
{context}

Rules:
1. Use ONLY the retrieved passages.
2. Do not use general knowledge.
3. Do not infer facts that are not stated.
4. Similar topic is not enough.
5. Return ONLY SUPPORTED or NOT_SUPPORTED.

SUPPORTED
NOT_SUPPORTED
"""

    response = model.invoke(
        [{"role": "user", "content": prompt}]
    )

    decision = response.content.strip().upper()

    return "SUPPORTED" in decision and "NOT_SUPPORTED" not in decision


def generate_answer(question: str, results) -> str:
    """Generate a grounded answer using only retrieved passages."""
    model = get_model()
    context = build_context(results)

    prompt = f"""
You are a technical documentation question-answering system.

Answer the user's question using ONLY the provided FastAPI
documentation passages.

USER QUESTION:
{question}

DOCUMENTATION PASSAGES:
{context}

Rules:
1. Use only the documentation passages.
2. Do not use general knowledge.
3. Do not invent details.
4. If the passages do not contain enough information, return:
   NOT_IN_SOURCE_MATERIAL
5. Cite supporting passages as [Source 1], [Source 2], etc.
6. Source numbers must match the numbered passages.
7. If code is provided, it must be supported by the passages.

Provide a concise technical answer.
"""

    response = model.invoke(
        [{"role": "user", "content": prompt}]
    )

    return response.content.strip()


def abstain(reason: str = ""):
    if reason:
        print(f"Abstention reason: {reason}")

    return {
        "answer": "NOT_IN_SOURCE_MATERIAL",
        "sources": [],
        "abstained": True,
        "reason": reason,
    }


def run_retrieval(question: str, threshold: float = RETRIEVAL_THRESHOLD):
    """
    Retrieval-only path used by evaluate.py.

    It never calls the LLM and therefore does not use Hugging Face
    inference credits.
    """
    results = retrieve_chunks(question, limit=TOP_K)

    if not results:
        return {
            "answer": "NOT_IN_SOURCE_MATERIAL",
            "sources": [],
            "abstained": True,
            "retrieval_score": None,
            "retrieval_pass": False,
            "reason": "No passages were retrieved.",
        }

    score = get_top_score(results)
    passed = check_retrieval_strength(results, threshold)

    return {
        "answer": "",
        "sources": get_sources(results),
        "abstained": not passed,
        "retrieval_score": score,
        "retrieval_pass": passed,
        "reason": (
            "Top retrieval score met threshold."
            if passed
            else "Top retrieval score was below threshold."
        ),
    }


def run_query(question: str, threshold: float = RETRIEVAL_THRESHOLD):
    """
    Full production path.

    This path may call Hugging Face and therefore can consume
    inference credits.
    """
    results = retrieve_chunks(question, limit=TOP_K)

    if not results:
        return abstain("No passages were retrieved.")

    if not check_retrieval_strength(results, threshold):
        return abstain(
            "Best retrieved passage was below the retrieval threshold."
        )

    if not check_answerability(question, results):
        return abstain(
            "Retrieved passages did not explicitly support the answer."
        )

    answer = generate_answer(question, results)

    if not answer or answer.upper() == "NOT_IN_SOURCE_MATERIAL":
        return abstain(
            "The answer generator could not produce a supported answer."
        )

    return {
        "answer": answer,
        "sources": get_sources(results),
        "abstained": False,
        "retrieval_score": get_top_score(results),
        "retrieval_pass": True,
    }


def main():
    print("=" * 70)
    print("FASTAPI CORPUS QA")
    print("=" * 70)
    print("Type 'exit' to quit.")

    while True:
        question = input("\nQuestion: ").strip()

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            break

        try:
            result = run_query(question)

            print("\nANSWER:")
            print(result["answer"])

            print("\nSOURCES:")
            for source in result.get("sources", []):
                print(
                    f"[Source {source['source_number']}] "
                    f"{source['url']} "
                    f"(score={source['score']:.4f})"
                )

        except Exception as exc:
            print(f"\nERROR: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
