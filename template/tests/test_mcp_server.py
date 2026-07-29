"""Smoke-test the MCP tool layer (mcp_server) end to end against the small corpus.

The tools open their own connection via ``_con()`` off the global settings singleton, so we point
``mcp_server.get_settings`` at the hermetic test DB and then call the five tools directly.
"""

from __future__ import annotations

import pytest

from research_kb import mcp_server
from research_kb.config import Settings
from research_kb.index import index_corpus


@pytest.fixture
def indexed(small_corpus: Settings, monkeypatch: pytest.MonkeyPatch) -> Settings:
    index_corpus(small_corpus)
    monkeypatch.setattr(mcp_server, "get_settings", lambda: small_corpus)
    return small_corpus


def test_kb_search_returns_hits_with_provenance(indexed: Settings):
    hits = mcp_server.kb_search("signal denoising measured series", k=5)
    assert hits
    for key in ("chunk_id", "document_id", "paper", "snippet", "score", "content_kind"):
        assert key in hits[0]


def test_kb_search_accepts_filters(indexed: Settings):
    assert isinstance(mcp_server.kb_search("denoising", k=5, tier="breadth"), list)


def test_kb_get_paper_and_missing(indexed: Settings):
    paper = mcp_server.kb_get_paper("Alpha Method")
    assert paper is not None
    assert paper["title"].startswith("Alpha")
    assert "section_outline" in paper and "artifacts" in paper
    assert mcp_server.kb_get_paper("no such document at all") is None


def test_kb_get_context(indexed: Settings):
    hits = mcp_server.kb_search("signal denoising", k=5)
    ctx = mcp_server.kb_get_context(hits[0]["chunk_id"])
    assert ctx is not None and "chunk" in ctx


def test_kb_follow_citations(indexed: Settings):
    alpha = mcp_server.kb_get_paper("Alpha Method")
    out = mcp_server.kb_follow_citations(alpha["id"], "out")
    assert out is not None and "edges" in out
    assert any(e.get("resolved") for e in out["edges"])  # Alpha's [1] resolves to Beta


def test_kb_list_corpus(indexed: Settings):
    corpus = mcp_server.kb_list_corpus(include_acquisition=True)
    assert corpus["count"] == 2
    assert "documents" in corpus and "acquisition_targets" in corpus
