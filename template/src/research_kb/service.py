"""Core service functions shared by the CLI and the MCP server.

Each returns plain JSON-serializable data so the MCP tools can return it directly and the CLI can
format it. This is the single implementation of every KB operation; both surfaces are thin wrappers.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from .citations import acquisition_targets
from .config import Settings, get_settings
from .embed.base import EmbeddingProvider
from .search import hybrid_search
from .store import (
    find_document_by_title,
    get_chunk,
    get_document,
    get_neighbor_chunks,
    list_documents,
)


def _doc_summary(con: sqlite3.Connection, doc_id: int) -> dict[str, Any]:
    doc = get_document(con, doc_id)
    return {"id": doc_id, "title": doc.title if doc else "?", "source_path": doc.source_path if doc else None}


def search_service(
    con: sqlite3.Connection,
    query: str,
    filters: dict[str, Any] | None = None,
    k: int = 10,
    embedder: EmbeddingProvider | None = None,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    hits = hybrid_search(con, query, embedder=embedder, filters=filters, k=k, settings=settings)
    return [h.model_dump() for h in hits]


def get_paper_service(con: sqlite3.Connection, identifier: str | int) -> dict[str, Any] | None:
    """Metadata + section outline + artifact links for a document (kb_get_paper)."""
    settings = get_settings()
    doc = None
    if isinstance(identifier, int) or str(identifier).isdigit():
        doc = get_document(con, int(identifier))
    if doc is None:
        doc = find_document_by_title(con, str(identifier))
    if doc is None or doc.id is None:
        return None

    outline_rows = con.execute(
        """
        SELECT section_number, section_title, page_start, chunk_index
        FROM chunks WHERE document_id = ? AND chunk_type = 'heading'
        ORDER BY chunk_index
        """,
        (doc.id,),
    ).fetchall()
    outline = [
        {"section_number": r["section_number"], "section_title": r["section_title"], "page_start": r["page_start"]}
        for r in outline_rows
    ]

    from pathlib import Path

    stem = Path(doc.source_path).stem
    artifact_dir = settings.artifact_dir(stem)
    artifacts = {
        name: str(artifact_dir / name)
        for name in ("verbatim.md", "enriched.md", "divergence-report.md", "llm.md")
        if (artifact_dir / name).exists()
    }

    return {
        "id": doc.id, "title": doc.title, "source_path": doc.source_path, "doc_type": doc.doc_type,
        "tier": doc.tier, "phase": doc.phase, "year": doc.year, "facet_a": doc.facet_a,
        "facet_b": doc.facet_b, "page_count": doc.page_count, "word_count": doc.word_count,
        "validated": doc.validated, "section_outline": outline, "artifacts": artifacts,
    }


def get_context_service(con: sqlite3.Connection, chunk_id: int) -> dict[str, Any] | None:
    """Parent section + neighboring chunks around a hit (kb_get_context)."""
    chunk = get_chunk(con, chunk_id)
    if chunk is None:
        return None
    parent = get_chunk(con, chunk.parent_chunk_id) if chunk.parent_chunk_id else None
    neighbors = get_neighbor_chunks(con, chunk, window=1)
    return {
        "chunk": chunk.model_dump(),
        "parent_section": (
            {
                "chunk_id": parent.id, "section_number": parent.section_number,
                "section_title": parent.section_title, "content": parent.content,
                "page_start": parent.page_start, "page_end": parent.page_end,
            }
            if parent
            else None
        ),
        "neighbors": [
            {"chunk_id": n.id, "chunk_index": n.chunk_index, "snippet": n.content[:200], "page_start": n.page_start}
            for n in neighbors
            if n.id != chunk_id
        ],
    }


def follow_citations_service(
    con: sqlite3.Connection, document_id: int, direction: str = "out"
) -> dict[str, Any] | None:
    """Traverse citation edges (kb_follow_citations). direction: 'out' | 'in'."""
    doc = get_document(con, document_id)
    if doc is None:
        return None
    if direction == "in":
        rows = con.execute(
            "SELECT id, from_document_id, context_snippet, page_ref FROM citations WHERE to_document_id = ?",
            (document_id,),
        ).fetchall()
        edges = [
            {
                "from": _doc_summary(con, r["from_document_id"]),
                "context": r["context_snippet"], "page_ref": r["page_ref"],
            }
            for r in rows
        ]
    else:
        rows = con.execute(
            "SELECT to_reference, to_document_id, page_ref FROM citations WHERE from_document_id = ? ORDER BY id",
            (document_id,),
        ).fetchall()
        edges = [
            {
                "to_reference": r["to_reference"],
                "resolved": _doc_summary(con, r["to_document_id"]) if r["to_document_id"] else None,
                "page_ref": r["page_ref"],
            }
            for r in rows
        ]
    return {"document": _doc_summary(con, document_id), "direction": direction, "edges": edges}


def list_corpus_service(
    con: sqlite3.Connection,
    filters: dict[str, Any] | None = None,
    include_acquisition: bool = False,
) -> dict[str, Any]:
    """What's indexed, optionally with cited-but-missing acquisition targets (kb_list_corpus)."""
    filters = filters or {}
    docs = list_documents(con, doc_type=filters.get("doc_type"), tier=filters.get("tier"), phase=filters.get("phase"))
    documents = []
    for d in docs:
        chunk_count = con.execute("SELECT COUNT(*) FROM chunks WHERE document_id = ?", (d.id,)).fetchone()[0]
        documents.append(
            {
                "id": d.id, "title": d.title, "source_path": d.source_path, "doc_type": d.doc_type,
                "tier": d.tier, "phase": d.phase, "facet_a": d.facet_a, "facet_b": d.facet_b,
                "validated": d.validated, "chunks": chunk_count,
            }
        )
    result: dict[str, Any] = {"count": len(documents), "documents": documents}
    if include_acquisition:
        result["acquisition_targets"] = acquisition_targets(con, limit=20)
    return result
