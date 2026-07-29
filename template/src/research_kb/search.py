"""Hybrid search: semantic (sqlite-vec) + keyword (FTS5 BM25), fused with RRF.

Semantic handles paraphrase/analogy; BM25 handles exact technical terms, which carry much of the
load for a vocabulary-dense corpus. Only leaf/derived chunks (``embedded = 1``) are searched;
parents are returned as context via :mod:`research_kb.store`. Metadata filters become SQL ``WHERE``
clauses; multi-term queries (``a | b | c``) fan out on the semantic side and OR-join on keyword.
"""

from __future__ import annotations

import re
import sqlite3

from .config import Settings, get_settings
from .db import pack_vector
from .embed import get_query_embedder
from .embed.base import EmbeddingProvider
from .models import SearchHit

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")
_FTS_STOP = frozenset({"the", "a", "an", "of", "to", "in", "on", "is", "and", "or", "how", "does", "what", "where"})
_FILTER_COLS = ("doc_type", "tier", "phase")


def _json_path(name: str) -> str:
    """SQLite JSON path for a facet name, with the name double-quote-escaped (bound as a parameter)."""
    return '$."' + name.replace('"', '""') + '"'


def _filter_sql(filters: dict[str, object] | None, alias: str = "d") -> tuple[str, list[object]]:
    """Build the WHERE fragment for scalar filters (doc_type/tier/phase) and named facets.

    ``filters["facets"]`` is a mapping ``{facet_name: value | [values]}``; a document matches when the
    value appears in that facet's array on its ``facets`` JSON column. json_each gives an exact
    array-membership test (no cross-facet false positives), and COALESCE tolerates a NULL column.
    """
    if not filters:
        return "", []
    clauses: list[str] = []
    params: list[object] = []
    for col in _FILTER_COLS:
        if filters.get(col) is not None:
            clauses.append(f"{alias}.{col} = ?")
            params.append(filters[col])
    facets = filters.get("facets")
    if isinstance(facets, dict):
        for name, values in facets.items():
            for v in values if isinstance(values, list) else [values]:
                clauses.append(
                    f"EXISTS (SELECT 1 FROM json_each(COALESCE({alias}.facets, '{{}}'), ?) je "
                    f"WHERE je.value = ?)"
                )
                params.append(_json_path(str(name)))
                params.append(v)
    return (" AND " + " AND ".join(clauses)) if clauses else "", params


def _allowed_ids(con: sqlite3.Connection, filters: dict[str, object] | None) -> set[int] | None:
    """Chunk ids passing metadata filters, or None when no filter is active (all allowed)."""
    if not filters:
        return None
    where, params = _filter_sql(filters)
    rows = con.execute(
        f"SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id WHERE c.embedded = 1{where}",
        params,
    ).fetchall()
    return {int(r["id"]) for r in rows}


def _fts_query(query: str) -> str | None:
    words: list[str] = []
    for term in query.split("|"):
        for w in _WORD_RE.findall(term):
            lw = w.lower()
            if len(lw) >= 2 and lw not in _FTS_STOP:
                words.append(f'"{lw}"')
    return " OR ".join(dict.fromkeys(words)) or None


def vec_search(
    con: sqlite3.Connection, query_vectors: list[list[float]], k: int, allowed: set[int] | None
) -> list[int]:
    """kNN over one or more query vectors, merged by best (smallest) distance; returns ranked ids."""
    pool = max(k * 4, 100)
    best: dict[int, float] = {}
    for qv in query_vectors:
        rows = con.execute(
            "SELECT id, distance FROM vec_chunks WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (pack_vector(qv), pool),
        ).fetchall()
        for r in rows:
            cid, dist = int(r["id"]), float(r["distance"])
            if allowed is not None and cid not in allowed:
                continue
            if cid not in best or dist < best[cid]:
                best[cid] = dist
    return [cid for cid, _ in sorted(best.items(), key=lambda kv: kv[1])][: pool]


def fts_search(con: sqlite3.Connection, query: str, k: int, filters: dict[str, object] | None) -> list[int]:
    match = _fts_query(query)
    if not match:
        return []
    where, params = _filter_sql(filters)
    rows = con.execute(
        f"""
        SELECT c.id AS id, bm25(chunks_fts) AS score
        FROM chunks_fts
        JOIN chunks c ON c.id = chunks_fts.rowid
        JOIN documents d ON d.id = c.document_id
        WHERE chunks_fts MATCH ? AND c.embedded = 1{where}
        ORDER BY score
        LIMIT ?
        """,
        [match, *params, max(k * 4, 100)],
    ).fetchall()
    return [int(r["id"]) for r in rows]


def reciprocal_rank_fusion(ranked_lists: list[list[int]], weights: list[float], rrf_k: int = 60) -> list[int]:
    """Fuse ranked id lists. Score = Σ weight / (rrf_k + rank)."""
    scores: dict[int, float] = {}
    for ids, w in zip(ranked_lists, weights, strict=True):
        for rank, cid in enumerate(ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + w / (rrf_k + rank)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)


def _hits_for_ids(con: sqlite3.Connection, ordered_ids: list[int], scores: dict[int, float]) -> list[SearchHit]:
    if not ordered_ids:
        return []
    placeholders = ",".join("?" * len(ordered_ids))
    rows = con.execute(
        f"""
        SELECT c.id, c.document_id, c.content, c.content_kind, c.chunk_type,
               c.section_number, c.section_title, c.page_start, c.page_end, d.title AS paper
        FROM chunks c JOIN documents d ON d.id = c.document_id
        WHERE c.id IN ({placeholders})
        """,
        ordered_ids,
    ).fetchall()
    by_id = {int(r["id"]): r for r in rows}
    hits: list[SearchHit] = []
    for cid in ordered_ids:
        r = by_id.get(cid)
        if r is None:
            continue
        snippet = r["content"].strip().replace("\n", " ")
        hits.append(
            SearchHit(
                chunk_id=cid, document_id=r["document_id"], paper=r["paper"],
                snippet=snippet[:400] + ("…" if len(snippet) > 400 else ""),
                section_number=r["section_number"], section_title=r["section_title"],
                page_start=r["page_start"], page_end=r["page_end"],
                content_kind=r["content_kind"], chunk_type=r["chunk_type"],
                score=round(scores.get(cid, 0.0), 6), retrieval="hybrid",
            )
        )
    return hits


def hybrid_search(
    con: sqlite3.Connection,
    query: str,
    embedder: EmbeddingProvider | None = None,
    filters: dict[str, object] | None = None,
    k: int = 10,
    settings: Settings | None = None,
) -> list[SearchHit]:
    """Hybrid semantic + keyword search fused with RRF."""
    settings = settings or get_settings()
    embedder = embedder or get_query_embedder(con, settings)
    allowed = _allowed_ids(con, filters)

    terms = [t.strip() for t in query.split("|") if t.strip()] or [query]
    query_vectors = embedder.embed_query(terms)
    semantic_ids = vec_search(con, query_vectors, k, allowed)
    keyword_ids = fts_search(con, query, k, filters)

    fused = reciprocal_rank_fusion(
        [semantic_ids, keyword_ids],
        [settings.semantic_weight, settings.keyword_weight],
        settings.rrf_k,
    )
    scores = {cid: 0.0 for cid in fused}
    for ids, w in ((semantic_ids, settings.semantic_weight), (keyword_ids, settings.keyword_weight)):
        for rank, cid in enumerate(ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + w / (settings.rrf_k + rank)

    return _hits_for_ids(con, fused[:k], scores)
