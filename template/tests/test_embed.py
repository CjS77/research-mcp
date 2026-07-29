"""The query embedder is self-describing: it follows the backend the DB was indexed with."""

from __future__ import annotations

from research_kb.config import Settings
from research_kb.db import connect
from research_kb.embed import get_query_embedder
from research_kb.index import index_corpus


def test_query_embedder_follows_db_meta(small_corpus: Settings):
    # small_corpus indexes with the hashing backend at dim 256; the query embedder must match that,
    # from the DB meta alone — a KB is searchable without re-supplying KB_EMBED_* config.
    index_corpus(small_corpus)
    con = connect(small_corpus.db_path)
    emb = get_query_embedder(con, small_corpus)
    assert emb.name == "hashing-v1"
    assert emb.dim == 256


def test_query_embedder_falls_back_to_config_without_meta(settings: Settings):
    from research_kb.db import init_db

    init_db(settings)
    con = connect(settings.db_path)
    con.execute("DELETE FROM kb_meta WHERE key = 'embed_backend'")
    con.commit()
    emb = get_query_embedder(con, settings)  # no backend meta -> config default (hashing, dim 256)
    assert emb.dim == settings.embed_dim
