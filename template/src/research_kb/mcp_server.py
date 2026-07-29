"""MCP server — the primary interface.

Exposes the KB as structured, filterable tools so it drops directly into Claude Code and other
research agents. Each tool is a thin wrapper over :mod:`research_kb.service` (the same core the CLI uses).
A fresh SQLite connection is opened per call (reads are fast; avoids cross-thread connection sharing).

Run with ``research-kb-mcp``.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .config import get_settings
from .db import connect
from .service import (
    follow_citations_service,
    get_context_service,
    get_paper_service,
    list_corpus_service,
    search_service,
)

mcp = FastMCP("research-kb")


def _con():
    return connect(get_settings().db_path)


@mcp.tool()
def kb_search(
    query: str,
    k: int = 10,
    doc_type: str | None = None,
    tier: str | None = None,
    phase: int | None = None,
    facet_a: str | None = None,
    facet_b: str | None = None,
) -> list[dict]:
    """Hybrid semantic + keyword search over the corpus.

    Supports multi-term queries split on ``|`` (e.g. ``"index structure | cache eviction | latency"``).
    Returns ranked hits with provenance: {chunk_id, paper, section, page, content_kind, score}.
    Filter by doc_type ('paper'|'research'|'assessment'|'sketch'|'spec'), tier ('core'|'breadth'),
    phase (1/2/3), facet_a, or facet_b.
    """
    filters: dict[str, object] = {}
    if doc_type:
        filters["doc_type"] = doc_type
    if tier:
        filters["tier"] = tier
    if phase is not None:
        filters["phase"] = phase
    if facet_a:
        filters["facet_a"] = [facet_a]
    if facet_b:
        filters["facet_b"] = [facet_b]
    con = _con()
    try:
        return search_service(con, query, filters=filters or None, k=k)
    finally:
        con.close()


@mcp.tool()
def kb_get_paper(identifier: str) -> dict | None:
    """Fetch a document's metadata, section outline, and distillation-artifact paths.

    ``identifier`` is either the numeric document id or a (partial) title.
    """
    con = _con()
    try:
        return get_paper_service(con, identifier)
    finally:
        con.close()


@mcp.tool()
def kb_get_context(chunk_id: int) -> dict | None:
    """Return the parent section and neighboring chunks for a search hit (expand a result in context)."""
    con = _con()
    try:
        return get_context_service(con, chunk_id)
    finally:
        con.close()


@mcp.tool()
def kb_follow_citations(document_id: int, direction: str = "out") -> dict | None:
    """Traverse the citation graph. direction='out' = works this paper cites (resolved + raw);
    direction='in' = corpus papers that cite this one."""
    con = _con()
    try:
        return follow_citations_service(con, document_id, direction)
    finally:
        con.close()


@mcp.tool()
def kb_list_corpus(
    tier: str | None = None,
    doc_type: str | None = None,
    phase: int | None = None,
    include_acquisition: bool = False,
) -> dict:
    """List what is indexed. Set ``include_acquisition`` to also return the ranked
    cited-but-missing works to acquire next."""
    filters: dict[str, object] = {}
    if tier:
        filters["tier"] = tier
    if doc_type:
        filters["doc_type"] = doc_type
    if phase is not None:
        filters["phase"] = phase
    con = _con()
    try:
        return list_corpus_service(con, filters=filters or None, include_acquisition=include_acquisition)
    finally:
        con.close()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
