"""Data-access layer: typed reads/writes over the SQLite tables.

Centralises row<->model marshalling (JSON array columns, provenance fields) so the indexer,
search, eval, and MCP layers never touch raw SQL rows.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from .db import pack_vector
from .models import Chunk, Citation, Document, EvalQuery


def _rowid(cur: sqlite3.Cursor) -> int:
    """Row id of the row the preceding INSERT created (always set after an INSERT)."""
    assert cur.lastrowid is not None
    return cur.lastrowid


def _dumps(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False)


def _loads(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        return [str(x) for x in parsed] if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def _now() -> str:
    return datetime.now(UTC).isoformat()


# --- Documents -----------------------------------------------------------------


def row_to_document(row: sqlite3.Row) -> Document:
    return Document(
        id=row["id"],
        source_path=row["source_path"],
        doc_type=row["doc_type"],
        tier=row["tier"],
        title=row["title"],
        phase=row["phase"],
        facet_b=_loads(row["facet_b"]),
        facet_a=_loads(row["facet_a"]),
        authors=_loads(row["authors"]),
        year=row["year"],
        page_count=row["page_count"],
        validated=bool(row["validated"]),
        content_hash=row["content_hash"],
        word_count=row["word_count"],
    )


def upsert_document(con: sqlite3.Connection, doc: Document) -> int:
    """Insert or replace a document row by ``source_path``; returns its id."""
    cur = con.execute(
        """
        INSERT INTO documents
            (source_path, doc_type, tier, title, phase, facet_b, facet_a, authors,
             year, page_count, validated, content_hash, word_count, indexed_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(source_path) DO UPDATE SET
            doc_type=excluded.doc_type, tier=excluded.tier, title=excluded.title,
            phase=excluded.phase, facet_b=excluded.facet_b, facet_a=excluded.facet_a,
            authors=excluded.authors, year=excluded.year, page_count=excluded.page_count,
            validated=excluded.validated, content_hash=excluded.content_hash,
            word_count=excluded.word_count, indexed_at=excluded.indexed_at
        """,
        (
            doc.source_path, doc.doc_type, doc.tier, doc.title, doc.phase,
            _dumps(doc.facet_b), _dumps(doc.facet_a), _dumps(doc.authors),
            doc.year, doc.page_count, int(doc.validated), doc.content_hash, doc.word_count, _now(),
        ),
    )
    if cur.lastrowid:
        row = con.execute("SELECT id FROM documents WHERE source_path = ?", (doc.source_path,)).fetchone()
        return int(row["id"])
    row = con.execute("SELECT id FROM documents WHERE source_path = ?", (doc.source_path,)).fetchone()
    return int(row["id"])


def get_document(con: sqlite3.Connection, doc_id: int) -> Document | None:
    row = con.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    return row_to_document(row) if row else None


def get_document_by_path(con: sqlite3.Connection, source_path: str) -> Document | None:
    row = con.execute("SELECT * FROM documents WHERE source_path = ?", (source_path,)).fetchone()
    return row_to_document(row) if row else None


def find_document_by_title(con: sqlite3.Connection, title: str) -> Document | None:
    """Resolve a title, preferring exact → prefix → substring matches (shortest title wins ties)."""
    for pattern in (title, f"{title}%", f"%{title}%"):
        row = con.execute(
            "SELECT * FROM documents WHERE title LIKE ? COLLATE NOCASE ORDER BY length(title) LIMIT 1",
            (pattern,),
        ).fetchone()
        if row:
            return row_to_document(row)
    return None


def list_documents(con: sqlite3.Connection, **filters: object) -> list[Document]:
    clauses, params = [], []
    for col in ("doc_type", "tier", "phase"):
        if filters.get(col) is not None:
            clauses.append(f"{col} = ?")
            params.append(filters[col])
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = con.execute(f"SELECT * FROM documents {where} ORDER BY doc_type, source_path", params).fetchall()
    return [row_to_document(r) for r in rows]


def set_validated(con: sqlite3.Connection, doc_id: int, validated: bool) -> None:
    con.execute(
        "UPDATE documents SET validated = ?, validated_at = ? WHERE id = ?",
        (int(validated), _now() if validated else None, doc_id),
    )


# --- Chunks --------------------------------------------------------------------


def row_to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        id=row["id"],
        document_id=row["document_id"],
        chunk_index=row["chunk_index"],
        content=row["content"],
        content_kind=row["content_kind"],
        section_number=row["section_number"],
        section_title=row["section_title"],
        chunk_type=row["chunk_type"],
        page_start=row["page_start"],
        page_end=row["page_end"],
        verbatim_hash=row["verbatim_hash"],
        parent_chunk_id=row["parent_chunk_id"],
        token_count=row["token_count"],
        embed_input=row["embed_input"],
        embedded=bool(row["embedded"]),
    )


def insert_chunk(con: sqlite3.Connection, chunk: Chunk) -> int:
    cur = con.execute(
        """
        INSERT INTO chunks
            (document_id, chunk_index, content, content_kind, section_number, section_title,
             chunk_type, page_start, page_end, verbatim_hash, parent_chunk_id, token_count,
             embed_input, embedded)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            chunk.document_id, chunk.chunk_index, chunk.content, chunk.content_kind,
            chunk.section_number, chunk.section_title, chunk.chunk_type, chunk.page_start,
            chunk.page_end, chunk.verbatim_hash, chunk.parent_chunk_id, chunk.token_count,
            chunk.embed_input, int(chunk.embedded),
        ),
    )
    return _rowid(cur)


def set_chunk_embedded(con: sqlite3.Connection, chunk_id: int, vector: list[float]) -> None:
    """Store a chunk's vector in vec_chunks and mark it embedded."""
    con.execute("INSERT OR REPLACE INTO vec_chunks(id, embedding) VALUES (?, ?)", (chunk_id, pack_vector(vector)))
    con.execute("UPDATE chunks SET embedded = 1 WHERE id = ?", (chunk_id,))


def get_chunk(con: sqlite3.Connection, chunk_id: int) -> Chunk | None:
    row = con.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
    return row_to_chunk(row) if row else None


def get_chunks_for_document(con: sqlite3.Connection, document_id: int) -> list[Chunk]:
    rows = con.execute(
        "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index", (document_id,)
    ).fetchall()
    return [row_to_chunk(r) for r in rows]


def get_child_chunks(con: sqlite3.Connection, parent_chunk_id: int) -> list[Chunk]:
    rows = con.execute(
        "SELECT * FROM chunks WHERE parent_chunk_id = ? ORDER BY chunk_index", (parent_chunk_id,)
    ).fetchall()
    return [row_to_chunk(r) for r in rows]


def get_neighbor_chunks(con: sqlite3.Connection, chunk: Chunk, window: int = 1) -> list[Chunk]:
    rows = con.execute(
        """
        SELECT * FROM chunks
        WHERE document_id = ? AND chunk_index BETWEEN ? AND ? AND parent_chunk_id IS NOT NULL
        ORDER BY chunk_index
        """,
        (chunk.document_id, chunk.chunk_index - window, chunk.chunk_index + window),
    ).fetchall()
    return [row_to_chunk(r) for r in rows]


def delete_document_data(con: sqlite3.Connection, document_id: int) -> None:
    """Remove a document's chunks (+ their vectors) and citations before re-indexing."""
    ids = [r["id"] for r in con.execute("SELECT id FROM chunks WHERE document_id = ?", (document_id,))]
    for cid in ids:
        con.execute("DELETE FROM vec_chunks WHERE id = ?", (cid,))
    con.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
    con.execute("DELETE FROM citations WHERE from_document_id = ?", (document_id,))


# --- Citations -----------------------------------------------------------------


def insert_citation(con: sqlite3.Connection, cit: Citation) -> int:
    cur = con.execute(
        """
        INSERT INTO citations (from_document_id, to_reference, to_document_id, context_snippet, page_ref)
        VALUES (?,?,?,?,?)
        """,
        (cit.from_document_id, cit.to_reference, cit.to_document_id, cit.context_snippet, cit.page_ref),
    )
    return _rowid(cur)


# --- Eval queries --------------------------------------------------------------


def clear_eval_queries(con: sqlite3.Connection) -> None:
    con.execute("DELETE FROM eval_queries")


def insert_eval_query(con: sqlite3.Connection, q: EvalQuery) -> int:
    cur = con.execute(
        """
        INSERT INTO eval_queries (query, expected_document_id, expected_section, expected_page, notes)
        VALUES (?,?,?,?,?)
        """,
        (q.query, q.expected_document_id, q.expected_section, q.expected_page, q.notes),
    )
    return _rowid(cur)


def list_eval_queries(con: sqlite3.Connection) -> list[EvalQuery]:
    rows = con.execute("SELECT * FROM eval_queries ORDER BY id").fetchall()
    return [
        EvalQuery(
            id=r["id"], query=r["query"], expected_document_id=r["expected_document_id"],
            expected_section=r["expected_section"], expected_page=r["expected_page"], notes=r["notes"],
        )
        for r in rows
    ]
