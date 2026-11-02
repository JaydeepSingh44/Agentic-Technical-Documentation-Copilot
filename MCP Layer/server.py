"""Documentation MCP server. Run with: python -m mcp_layer.server"""
import os
import socket
import sys
from contextlib import redirect_stdout
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from mcp.server.fastmcp import FastMCP

from human_approval.approval import create_proposal, get_proposal, list_proposals
from human_approval.approval import mark_ingested
# Existing modules print initialization diagnostics; MCP stdio reserves stdout
# for protocol frames, so redirect those diagnostics to stderr.
with redirect_stdout(sys.stderr):
    from ingest2 import ingest_url
    from query2 import retrieve_chunks, get_sources

mcp = FastMCP("Technical Documentation Copilot")
ALLOWED_DOMAINS = [
    d.strip().lower() for d in os.getenv(
        "ALLOWED_DOC_DOMAINS", "fastapi.tiangolo.com,docs.langchain.com,langchain-ai.github.io, qdrant.tech,modelcontextprotocol.io"
    ).split(",") if d.strip()
]


def _validate_official_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("Only valid HTTPS documentation URLs are accepted")
    host = parsed.hostname.lower().rstrip(".")
    if host in {"localhost"} or host.endswith(".local"):
        raise ValueError("Local URLs are not allowed")
    try:
        for info in socket.getaddrinfo(host, None):
            ip = info[4][0]
            if ip.startswith(("127.", "10.", "192.168.", "169.254.")) or ip == "::1" or ip.startswith("fc") or ip.startswith("fd"):
                raise ValueError("Private or local network destinations are not allowed")
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve documentation host: {host}") from exc
    if not any(host == domain or host.endswith("." + domain) for domain in ALLOWED_DOMAINS):
        raise ValueError(f"Domain '{host}' is not allowlisted. Configure ALLOWED_DOC_DOMAINS in .env")
    return url


def _fetch_page(url: str) -> tuple[str, str]:
    safe_url = _validate_official_url(url)
    response = requests.get(
        safe_url, timeout=20, allow_redirects=True,
        headers={"User-Agent": "TechnicalDocumentationCopilot/1.0"},
    )
    response.raise_for_status()
    _validate_official_url(response.url)
    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else safe_url
    for node in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
        node.decompose()
    main = soup.find("main") or soup.body or soup
    text = main.get_text("\n", strip=True)
    return title, text


@mcp.tool()
def search_documentation(query: str, limit: int = 5) -> dict:
    """Search the already indexed official technical documentation in Qdrant."""
    if not query.strip():
        raise ValueError("query cannot be empty")
    points = retrieve_chunks(query, limit=max(1, min(limit, 10)))
    return {"query": query, "count": len(points), "results": get_sources(points)}


@mcp.tool()
def discover_documentation(technology: str, topic: str, max_results: int = 5) -> dict:
    """Find candidate documentation pages; only allowlisted official domains are returned."""
    search_query = f"{technology} official documentation {topic}"
    found = []
    with DDGS() as ddgs:
        for result in ddgs.text(search_query, max_results=max(1, min(max_results * 3, 15))):
            url = result.get("href") or result.get("url")
            if not url:
                continue
            try:
                _validate_official_url(url)
            except Exception:
                continue
            found.append({"title": result.get("title", ""), "url": url, "snippet": result.get("body", "")})
            if len(found) >= max_results:
                break
    return {"search_query": search_query, "results": found, "count": len(found)}


@mcp.tool()
def prepare_documentation(url: str) -> dict:
    """Fetch an official documentation page and create a pending human-review proposal. Does not ingest it."""
    title, text = _fetch_page(url)
    if len(text.strip()) < 100:
        raise ValueError("Page did not contain enough readable documentation text")
    proposal = create_proposal(url, title, text)
    return {
        "proposal_id": proposal["proposal_id"], "title": title, "url": url,
        "status": "pending_human_approval", "preview": text[:1000],
        "message": "No Qdrant write has occurred. A human must approve this proposal first.",
    }


@mcp.tool()
def list_pending_documentation() -> dict:
    """List documentation proposals waiting for a human decision."""
    return {"proposals": list_proposals("pending")}


@mcp.tool()
def ingest_approved_documentation(proposal_id: str) -> dict:
    """Ingest a proposal only if a human has approved it through the local approval CLI/UI."""
    proposal = get_proposal(proposal_id)
    if proposal.get("status") != "approved":
        raise PermissionError("Human approval is required before ingestion")
    with redirect_stdout(sys.stderr):
        result = ingest_url(proposal["url"])
    mark_ingested(proposal_id)
    return {"status": "ingested", "proposal_id": proposal_id, "result": result}


if __name__ == "__main__":
    mcp.run(transport="stdio")
