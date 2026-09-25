import os
import re
import hashlib
from pathlib import Path
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


if not HF_TOKEN:
    raise ValueError("HF_TOKEN is not set in .env")

if not QDRANT_URL:
    raise ValueError("QDRANT_URL is not set in .env")

if not QDRANT_API_KEY:
    raise ValueError("QDRANT_API_KEY is not set in .env")


# ============================================================
# CONFIGURATION
# ============================================================

COLLECTION_NAME = "fastapi_corpus"

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

RECURSIVE_CHUNK_SIZE = 1500
RECURSIVE_CHUNK_OVERLAP = 150

CACHE_DIR = Path("data/fastapi_raw")

REQUEST_TIMEOUT = 30

BATCH_SIZE = 32


# ============================================================
# FASTAPI CORPUS
# ============================================================

FASTAPI_URLS = [

    # --------------------------------------------------------
    # Basic concepts
    # --------------------------------------------------------

    "https://fastapi.tiangolo.com/tutorial/first-steps/",

    "https://fastapi.tiangolo.com/tutorial/path-params/",

    "https://fastapi.tiangolo.com/tutorial/query-params/",

    "https://fastapi.tiangolo.com/tutorial/query-params-str-validations/",

    "https://fastapi.tiangolo.com/tutorial/path-params-numeric-validations/",

    "https://fastapi.tiangolo.com/tutorial/query-param-models/",


    # --------------------------------------------------------
    # Request bodies
    # --------------------------------------------------------

    "https://fastapi.tiangolo.com/tutorial/body/",

    "https://fastapi.tiangolo.com/tutorial/body-multiple-params/",

    "https://fastapi.tiangolo.com/tutorial/body-fields/",

    "https://fastapi.tiangolo.com/tutorial/body-nested-models/",


    # --------------------------------------------------------
    # Responses
    # --------------------------------------------------------

    "https://fastapi.tiangolo.com/tutorial/response-model/",

    "https://fastapi.tiangolo.com/tutorial/response-status-code/",


    # --------------------------------------------------------
    # Forms / files
    # --------------------------------------------------------

    "https://fastapi.tiangolo.com/tutorial/request-forms/",

    "https://fastapi.tiangolo.com/tutorial/request-files/",

    "https://fastapi.tiangolo.com/tutorial/request-forms-and-files/",


    # --------------------------------------------------------
    # Errors / configuration
    # --------------------------------------------------------

    "https://fastapi.tiangolo.com/tutorial/handling-errors/",

    "https://fastapi.tiangolo.com/tutorial/path-operation-configuration/",

    "https://fastapi.tiangolo.com/tutorial/encoder/",


    # --------------------------------------------------------
    # Dependencies
    # --------------------------------------------------------

    "https://fastapi.tiangolo.com/tutorial/dependencies/",

    "https://fastapi.tiangolo.com/tutorial/dependencies/classes-as-dependencies/",

    "https://fastapi.tiangolo.com/tutorial/dependencies/sub-dependencies/",

    "https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-in-path-operation-decorators/",

    "https://fastapi.tiangolo.com/tutorial/dependencies/global-dependencies/",


    # --------------------------------------------------------
    # Security
    # --------------------------------------------------------

    "https://fastapi.tiangolo.com/tutorial/security/",

    "https://fastapi.tiangolo.com/tutorial/security/simple-oauth2/",


    # --------------------------------------------------------
    # Middleware / CORS
    # --------------------------------------------------------

    "https://fastapi.tiangolo.com/tutorial/middleware/",

    "https://fastapi.tiangolo.com/tutorial/cors/",


    # --------------------------------------------------------
    # Background tasks
    # --------------------------------------------------------

    "https://fastapi.tiangolo.com/tutorial/background-tasks/",


    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    "https://fastapi.tiangolo.com/tutorial/metadata/",

    "https://fastapi.tiangolo.com/reference/parameters/",
]


# ============================================================
# EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embedding_model = HuggingFaceEndpointEmbeddings(
    model=EMBEDDING_MODEL,
    huggingfacehub_api_token=HF_TOKEN,
)

print("Embedding model loaded.")


# ============================================================
# QDRANT CLIENT
# ============================================================

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60,
)


# ============================================================
# CACHE DIRECTORY
# ============================================================

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DOCUMENT ID
# ============================================================

def create_document_id(url: str) -> str:

    return hashlib.sha256(
        url.encode("utf-8")
    ).hexdigest()[:16]


# ============================================================
# CHUNK ID
# ============================================================

def create_chunk_id(
    document_id: str,
    chunk_text: str,
) -> str:

    raw = f"{document_id}:{chunk_text}"

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:32]


# ============================================================
# CACHE FILE NAME
# ============================================================

def cache_filename(url: str) -> Path:

    document_id = create_document_id(url)

    return CACHE_DIR / f"{document_id}.html"


# ============================================================
# LOAD HTML
# ============================================================

def load_html(url: str) -> str:

    cache_file = cache_filename(url)

    # --------------------------------------------------------
    # Use cached HTML if it already exists
    # --------------------------------------------------------

    if cache_file.exists():

        print("Using cached HTML:")
        print(cache_file)

        return cache_file.read_text(
            encoding="utf-8"
        )


    # --------------------------------------------------------
    # Download webpage
    # --------------------------------------------------------

    print("\nDownloading:")
    print(url)

    response = requests.get(
        url,
        timeout=REQUEST_TIMEOUT,
        headers={
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/140.0 Safari/537.36"
            )
        },
    )

    response.raise_for_status()

    print(
        f"HTTP status: {response.status_code}"
    )


    # --------------------------------------------------------
    # Save raw HTML
    # --------------------------------------------------------

    cache_file.write_text(
        response.text,
        encoding="utf-8",
    )

    print(
        f"Cached HTML: {cache_file}"
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


    # --------------------------------------------------------
    # Remove elements that are not documentation content
    # --------------------------------------------------------

    for element in soup(
        [
            "script",
            "style",
            "nav",
            "footer",
            "noscript",
            "svg",
            "header",
        ]
    ):

        element.decompose()


    # --------------------------------------------------------
    # Focus on documentation content when available
    # --------------------------------------------------------

    main = soup.find("main")

    if main:

        return str(main)


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
        f"Document sections: {len(sections)}"
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
        # Preserve heading context
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
        # Recursive fallback
        # ----------------------------------------------------

        for semantic_chunk in semantic_chunks:

            if len(semantic_chunk) > RECURSIVE_CHUNK_SIZE:

                recursive_chunks = (
                    recursive_splitter.split_text(
                        semantic_chunk
                    )
                )

                for chunk in recursive_chunks:

                    if chunk.strip():

                        final_chunks.append(
                            {
                                "text": chunk.strip(),
                                "heading": heading_path,
                                "h1": h1,
                                "h2": h2,
                                "h3": h3,
                            }
                        )

            else:

                if semantic_chunk.strip():

                    final_chunks.append(
                        {
                            "text": semantic_chunk.strip(),
                            "heading": heading_path,
                            "h1": h1,
                            "h2": h2,
                            "h3": h3,
                        }
                    )


    print(
        f"Final chunks: {len(final_chunks)}"
    )

    return final_chunks


# ============================================================
# CREATE QDRANT COLLECTION
# ============================================================

def create_collection(vector_size: int):

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
# INGEST ONE URL
# ============================================================

def ingest_url(url: str):

    print("\n")
    print("=" * 70)
    print("STARTING DOCUMENT INGESTION")
    print("=" * 70)

    print(f"URL: {url}")


    # --------------------------------------------------------
    # Document information
    # --------------------------------------------------------

    document_id = create_document_id(
        url
    )

    parsed_url = urlparse(url)

    domain = parsed_url.netloc


    print(
        f"Document ID: {document_id}"
    )

    print(
        f"Domain: {domain}"
    )


    # --------------------------------------------------------
    # Load HTML
    # --------------------------------------------------------

    html = load_html(url)


    # --------------------------------------------------------
    # Clean HTML
    # --------------------------------------------------------

    cleaned_html = clean_html(
        html
    )


    # --------------------------------------------------------
    # Split into documentation sections
    # --------------------------------------------------------

    sections = document_aware_split(
        cleaned_html
    )


    if not sections:

        raise ValueError(
            f"No sections found for {url}"
        )


    # --------------------------------------------------------
    # Create chunks
    # --------------------------------------------------------

    chunks = create_chunks(
        sections
    )


    if not chunks:

        raise ValueError(
            f"No chunks generated for {url}"
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
            f"No embeddings generated for {url}"
        )


    vector_size = len(
        vectors[0]
    )


    print(
        f"Vector size: {vector_size}"
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

            "corpus":
                "FastAPI official documentation",
        }


        points.append(
            PointStruct(
                id=chunk_id,

                vector=vector,

                payload=payload,
            )
        )


    # --------------------------------------------------------
    # Upload to Qdrant in batches
    # --------------------------------------------------------

    print(
        f"\nUploading {len(points)} chunks..."
    )


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

    print("\n" + "-" * 70)

    print(
        "DOCUMENT INGESTED"
    )

    print(
        f"URL: {url}"
    )

    print(
        f"Sections: {len(sections)}"
    )

    print(
        f"Chunks: {len(chunks)}"
    )

    print(
        "-" * 70
    )


    return {
        "url": url,
        "document_id": document_id,
        "sections": len(sections),
        "chunks": len(chunks),
    }


# ============================================================
# INGEST ALL FASTAPI DOCUMENTS
# ============================================================

def ingest_all():

    print("\n")
    print("=" * 70)
    print("FASTAPI CORPUS INGESTION")
    print("=" * 70)

    print(
        f"Documents to ingest: "
        f"{len(FASTAPI_URLS)}"
    )

    print(
        f"Collection: "
        f"{COLLECTION_NAME}"
    )

    print("=" * 70)


    successful = []

    failed = []


    for index, url in enumerate(
        FASTAPI_URLS,
        start=1,
    ):

        print("\n")
        print(
            f"[{index}/{len(FASTAPI_URLS)}]"
        )


        try:

            result = ingest_url(
                url
            )

            successful.append(
                result
            )


        except Exception as e:

            print(
                f"FAILED: {url}"
            )

            print(
                f"ERROR: {e}"
            )

            failed.append(
                {
                    "url": url,
                    "error": str(e),
                }
            )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("CORPUS INGESTION COMPLETE")
    print("=" * 70)


    print(
        f"Total documents: "
        f"{len(FASTAPI_URLS)}"
    )

    print(
        f"Successful: "
        f"{len(successful)}"
    )

    print(
        f"Failed: "
        f"{len(failed)}"
    )


    total_chunks = sum(
        item["chunks"]
        for item in successful
    )


    print(
        f"Total chunks: "
        f"{total_chunks}"
    )


    if failed:

        print("\nFAILED DOCUMENTS:")

        for item in failed:

            print(
                f"- {item['url']}"
            )

            print(
                f"  {item['error']}"
            )


    print("\n")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    ingest_all()