"""Embedding properties + search building blocks (RRF, FTS query, hybrid)."""

from __future__ import annotations

import math

from research_kb.embed.hashing import HashingEmbedder
from research_kb.search import _fts_query, reciprocal_rank_fusion


def test_hashing_embedder_deterministic_and_unit_norm():
    emb = HashingEmbedder(dim=256)
    a1 = emb.embed_one("time series forecasting")
    a2 = emb.embed_one("time series forecasting")
    assert a1 == a2
    assert len(a1) == 256
    assert math.sqrt(sum(x * x for x in a1)) == 1.0 or math.isclose(sum(x * x for x in a1), 1.0, rel_tol=1e-6)


def test_hashing_similarity_orders_sensibly():
    emb = HashingEmbedder(dim=512)
    q = emb.embed_one("distributed database query planning")
    near = emb.embed_one("distributed database query optimization")
    far = emb.embed_one("baroque chamber music of the eighteenth century")
    def dot(u, v):
        return sum(a * b for a, b in zip(u, v, strict=True))

    assert dot(q, near) > dot(q, far)


def test_rrf_prefers_items_in_both_lists():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 4, 5]], weights=[0.7, 0.3], rrf_k=60)
    assert fused[0] == 3  # only id present in both lists ranks first


def test_rrf_respects_weight_when_disjoint():
    fused = reciprocal_rank_fusion([[10], [20]], weights=[0.7, 0.3], rrf_k=60)
    assert fused[0] == 10  # higher-weighted list's top item wins


def test_fts_query_builds_or_expression_and_drops_stopwords():
    q = _fts_query("how does the cache resist a stale read")
    assert q is not None
    assert '"cache"' in q and " OR " in q
    assert '"the"' not in q and '"how"' not in q


def test_fts_query_handles_multiterm_pipes():
    q = _fts_query("cache | write-through policy | eviction")
    assert '"cache"' in q and '"eviction"' in q
