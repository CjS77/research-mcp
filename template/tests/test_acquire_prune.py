"""Verified acquisition + corpus pruning."""

from __future__ import annotations

from pathlib import Path

from research_kb.acquire import _sig_words, verify_pdf
from research_kb.config import Settings
from research_kb.db import connect
from research_kb.index import index_corpus


def test_verify_rejects_non_pdf(tmp_path: Path):
    p = tmp_path / "x.pdf"
    p.write_bytes(b"<!DOCTYPE html> not a pdf")
    ok, reason = verify_pdf(p, "Some Real Title Here")
    assert not ok and "PDF" in reason


def test_sig_words_drops_stopwords():
    assert _sig_words("The Analysis of Fast and Robust Estimators") == {"analysis", "fast", "robust", "estimators"}


def test_prune_removes_vanished_sources(small_corpus: Settings):
    index_corpus(small_corpus)
    # delete one source file, then re-index: its document + chunks must be pruned
    (small_corpus.reference_dir / "beta-denoising-method.md").unlink()
    summary = index_corpus(small_corpus)
    assert any("beta-denoising-method" in p for p in summary.pruned)

    con = connect(small_corpus.db_path)
    remaining = con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    assert remaining == 1
    orphan_chunks = con.execute(
        "SELECT COUNT(*) FROM chunks c LEFT JOIN documents d ON d.id = c.document_id WHERE d.id IS NULL"
    ).fetchone()[0]
    assert orphan_chunks == 0
