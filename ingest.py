import os
import hashlib
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    HTMLHeaderTextSplitter,
)

from langchain_huggingface import HuggingFaceEndpointEmbeddings


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


# ============================================================
# CONFIGURATION
# ============================================================

COLLECTION_NAME = "developer_docs"

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

RECURSIVE_CHUNK_SIZE = 1500
RECURSIVE_CHUNK_OVERLAP = 150


# ============================================================
# EMBEDDING MODEL
# ============================================================

embedding_model = HuggingFaceEndpointEmbeddings(
    model=EMBEDDING_MODEL,
    huggingfacehub_api_token=HF_TOKEN,
)


# ============================================================
# QDRANT CLIENT
# ============================================================

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60,
)


# ============================================================
# DOCUMENT ID
# ============================================================

def create_document_id(url: str):

    return hashlib.sha256(
        url.encode("utf-8")
    ).hexdigest()[:16]


# ============================================================
# CHUNK ID
# ============================================================

def create_chunk_id(
    document_id: str,
    chunk_text: str,
):

    raw = f"{document_id}:{chunk_text}"

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:32]


# ============================================================
# LOAD HTML
# ============================================================

def load_html(url: str) -> str:

    print("\nLoading URL:")
    print(url)

    response = requests.get(
        url,
        timeout=30,
        headers={
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "Chrome/140.0 Safari/537.36"
            )
        },
    )

    response.raise_for_status()

    print(
        f"HTTP status: "
        f"{response.status_code}"
    )

    return response.text


# ============================================================
# CLEAN HTML
# ============================================================

def clean_html(html: str) -> str:

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    for element in soup(
        [
            "script",
            "style",
            "nav",
            "footer",
            "noscript",
            "svg",
        ]
    ):
        element.decompose()

    return str(soup)


# ============================================================
# DOCUMENT-AWARE SPLITTING
# ============================================================

def document_aware_split(html: str):

    headers_to_split_on = [
        ("h1", "h1"),
        ("h2", "h2"),
        ("h3", "h3"),
    ]

    splitter = HTMLHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

    sections = splitter.split_text(
        html
    )

    print(
        f"Document sections: "
        f"{len(sections)}"
    )

    return sections


# ============================================================
# CREATE CHUNKS
# ============================================================

def create_chunks(sections):

    semantic_splitter = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90,
    )

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=RECURSIVE_CHUNK_SIZE,
        chunk_overlap=RECURSIVE_CHUNK_OVERLAP,
    )

    final_chunks = []

    for section in sections:

        metadata = section.metadata

        h1 = metadata.get(
            "h1",
            "",
        )

        h2 = metadata.get(
            "h2",
            "",
        )

        h3 = metadata.get(
            "h3",
            "",
        )

        heading_parts = [
            value
            for value in [
                h1,
                h2,
                h3,
            ]
            if value
        ]

        heading_path = " > ".join(
            heading_parts
        )

        section_text = (
            section.page_content.strip()
        )

        if not section_text:
            continue

        # ----------------------------------------------------
        # Add heading context to the chunk
        # ----------------------------------------------------

        if heading_path:

            contextual_text = (
                f"Documentation Section: "
                f"{heading_path}\n\n"
                f"{section_text}"
            )

        else:

            contextual_text = section_text


        # ----------------------------------------------------
        # Semantic chunking
        # ----------------------------------------------------

        semantic_chunks = (
            semantic_splitter.split_text(
                contextual_text
            )
        )


        # ----------------------------------------------------
        # Recursive fallback for large chunks
        # ----------------------------------------------------

        for semantic_chunk in semantic_chunks:

            if (
                len(semantic_chunk)
                > RECURSIVE_CHUNK_SIZE
            ):

                recursive_chunks = (
                    recursive_splitter.split_text(
                        semantic_chunk
                    )
                )

                for chunk in recursive_chunks:

                    final_chunks.append(
                        {
                            "text": chunk,
                            "heading": heading_path,
                            "h1": h1,
                            "h2": h2,
                            "h3": h3,
                        }
                    )

            else:

                final_chunks.append(
                    {
                        "text": semantic_chunk,
                        "heading": heading_path,
                        "h1": h1,
                        "h2": h2,
                        "h3": h3,
                    }
                )

    print(
        f"Final chunks: "
        f"{len(final_chunks)}"
    )

    return final_chunks


# ============================================================
# CREATE QDRANT COLLECTION
# ============================================================

def create_collection(
    vector_size: int,
):

    if not client.collection_exists(
        COLLECTION_NAME
    ):

        client.create_collection(
            collection_name=COLLECTION_NAME,

            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

        print(
            f"Created collection: "
            f"{COLLECTION_NAME}"
        )

    else:

        print(
            f"Collection already exists: "
            f"{COLLECTION_NAME}"
        )


# ============================================================
# INGEST URL
# ============================================================

def ingest_url(url: str):

    print("\n" + "=" * 60)
    print("STARTING INGESTION")
    print("=" * 60)


    # --------------------------------------------------------
    # Document information
    # --------------------------------------------------------

    document_id = create_document_id(
        url
    )

    parsed_url = urlparse(url)

    domain = parsed_url.netloc

    print(
        f"Document ID: "
        f"{document_id}"
    )

    print(
        f"Domain: "
        f"{domain}"
    )


    # --------------------------------------------------------
    # Load HTML
    # --------------------------------------------------------

    html = load_html(url)

    print(
        f"Raw HTML length: "
        f"{len(html)}"
    )


    # --------------------------------------------------------
    # Clean HTML
    # --------------------------------------------------------

    cleaned_html = clean_html(
        html
    )

    print(
        f"Clean HTML length: "
        f"{len(cleaned_html)}"
    )


    # --------------------------------------------------------
    # Document-aware splitting
    # --------------------------------------------------------

    sections = document_aware_split(
        cleaned_html
    )

    if not sections:

        raise ValueError(
            "No documentation sections found."
        )


    # --------------------------------------------------------
    # Create chunks
    # --------------------------------------------------------

    chunks = create_chunks(
        sections
    )

    if not chunks:

        raise ValueError(
            "No chunks generated."
        )


    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    print(
        "\nGenerating embeddings..."
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    vectors = (
        embedding_model.embed_documents(
            texts
        )
    )

    if not vectors:

        raise ValueError(
            "No embeddings generated."
        )


    vector_size = len(
        vectors[0]
    )

    print(
        f"Vector size: "
        f"{vector_size}"
    )


    # --------------------------------------------------------
    # Create Qdrant collection
    # --------------------------------------------------------

    create_collection(
        vector_size
    )


    # --------------------------------------------------------
    # Create Qdrant points
    # --------------------------------------------------------

    points = []

    for index, (
        chunk,
        vector,
    ) in enumerate(
        zip(
            chunks,
            vectors,
        )
    ):

        chunk_id = create_chunk_id(
            document_id,
            chunk["text"],
        )


        payload = {

            "text": chunk["text"],

            "url": url,

            "domain": domain,

            "document_id": document_id,

            "chunk_id": chunk_id,

            "chunk_index": index,

            "heading": chunk["heading"],

            "h1": chunk["h1"],

            "h2": chunk["h2"],

            "h3": chunk["h3"],

            "embedding_model":
                EMBEDDING_MODEL,
        }


        points.append(
            PointStruct(
                id=chunk_id,

                vector=vector,

                payload=payload,
            )
        )


    # --------------------------------------------------------
    # Upload to Qdrant
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Upload to Qdrant
    # --------------------------------------------------------

    print(
        f"\nUploading "
        f"{len(points)} chunks..."
    )

    BATCH_SIZE = 10

    for start in range(
        0,
        len(points),
        BATCH_SIZE,
    ):

        batch = points[
            start:start + BATCH_SIZE
        ]

        print(
            f"Uploading chunks "
            f"{start + 1}-"
            f"{start + len(batch)} "
            f"of {len(points)}..."
        )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch,
            wait=True,
        )


    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "INGESTION COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"URL: "
        f"{url}"
    )

    print(
        f"Document ID: "
        f"{document_id}"
    )

    print(
        f"Sections: "
        f"{len(sections)}"
    )

    print(
        f"Chunks: "
        f"{len(chunks)}"
    )

    print(
        f"Collection: "
        f"{COLLECTION_NAME}"
    )


    return {
        "url": url,
        "document_id": document_id,
        "sections": len(sections),
        "chunks": len(chunks),
    }

# ============================================================
# DIRECT INGESTION TEST
# ============================================================

if __name__ == "__main__":

    test_url = (
        "https://docs.langchain.com/"
        "oss/python/langchain/overview"
    )

    ingest_url(
        test_url
    )