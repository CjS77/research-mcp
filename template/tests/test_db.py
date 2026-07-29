"""DB layer: schema init, vector round-trip, dimension guard, meta."""

from __future__ import annotations

import pytest

from research_kb.config import Settings
from research_kb.db import connect, get_meta, init_db, pack_vector, unpack_vector


def test_init_creates_schema_and_vec_table(settings: Settings):
    con = init_db(settings)
    names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
    assert {"documents", "chunks", "chunks_fts", "citations", "eval_queries", "indexing_jobs", "vec_chunks"} <= names
    assert get_meta(con, "embed_dim") == str(settings.embed_dim)


def test_vector_pack_roundtrip():
    vec = [0.5, -0.25, 1.0, 0.0]
    assert unpack_vector(pack_vector(vec)) == pytest.approx(vec)


def test_vec_knn_roundtrip(settings: Settings):
    con = init_db(settings)
    con.execute("INSERT INTO documents(source_path, doc_type, tier, title) VALUES ('p', 'paper', 'core', 't')")
    con.execute(
        "INSERT INTO chunks(document_id, chunk_index, content, content_kind, chunk_type, embedded) "
        "VALUES (1, 0, 'x', 'verbatim', 'paragraph', 1)"
    )
    con.execute("INSERT INTO vec_chunks(id, embedding) VALUES (1, ?)", (pack_vector([1.0] + [0.0] * (settings.embed_dim - 1)),))
    rows = con.execute(
        "SELECT id, distance FROM vec_chunks WHERE embedding MATCH ? AND k = 1 ORDER BY distance",
        (pack_vector([1.0] + [0.0] * (settings.embed_dim - 1)),),
    ).fetchall()
    assert rows[0]["id"] == 1
    assert rows[0]["distance"] == pytest.approx(0.0, abs=1e-5)


def test_dimension_mismatch_raises(settings: Settings):
    init_db(settings).close()
    bad = settings.model_copy(update={"embed_dim": settings.embed_dim + 8})
    with pytest.raises(RuntimeError, match="dimension mismatch"):
        init_db(bad, con=connect(bad.db_path))
