"""End-to-end: index a tiny corpus, then search / cite / eval / service over it."""

from __future__ import annotations

from pathlib import Path

from research_kb.config import Settings
from research_kb.db import connect
from research_kb.eval import faithfulness_check, load_gold_queries, run_eval
from research_kb.index import index_corpus
from research_kb.service import (
    follow_citations_service,
    get_context_service,
    get_paper_service,
    list_corpus_service,
    search_service,
)


def test_index_search_cite_eval(small_corpus: Settings):
    summary = index_corpus(small_corpus)
    assert len(summary.indexed) == 2
    assert summary.chunks_created > 0
    assert not summary.failed

    con = connect(small_corpus.db_path)

    # Search finds the relevant doc.
    hits = search_service(con, "signal denoising measured series", k=5, settings=small_corpus)
    assert hits
    assert any("Beta" in h["paper"] or "Alpha" in h["paper"] for h in hits)

    # Alpha's [1] resolves to Beta (title n-gram); acquisition keeps the uncited external work.
    resolved = con.execute("SELECT COUNT(*) FROM citations WHERE to_document_id IS NOT NULL").fetchone()[0]
    assert resolved >= 1

    alpha = get_paper_service(con, "Alpha Method")
    assert alpha is not None and alpha["section_outline"]
    out = follow_citations_service(con, alpha["id"], "out")
    assert any(e["resolved"] for e in out["edges"])  # at least one resolved edge

    # Context expansion returns a parent section.
    ctx = get_context_service(con, hits[0]["chunk_id"])
    assert ctx is not None and "chunk" in ctx

    # Faithfulness: every verbatim chunk still hashes to its recorded verbatim_hash.
    checked, matched = faithfulness_check(con)
    assert checked > 0 and checked == matched

    corpus = list_corpus_service(con, include_acquisition=True)
    assert corpus["count"] == 2
    assert "acquisition_targets" in corpus


def test_incremental_skip(small_corpus: Settings):
    index_corpus(small_corpus)
    again = index_corpus(small_corpus)  # unchanged hashes → skipped
    assert len(again.skipped) == 2
    assert not again.indexed


def test_eval_harness_on_small_corpus(small_corpus: Settings, tmp_path: Path):
    index_corpus(small_corpus)
    con = connect(small_corpus.db_path)
    gold = tmp_path / "gold.yaml"
    gold.write_text(
        "- query: signal denoising with no reference template\n"
        "  expected_source_path: reference/beta-denoising-method.md\n",
        encoding="utf-8",
    )
    # expected_source_path is repo-relative in real use; here docs live under tmp, so patch the row.
    load_gold_queries(con, gold, small_corpus)
    # Repoint the expected doc to the actual indexed beta doc for a meaningful metric.
    beta = con.execute("SELECT id FROM documents WHERE title LIKE 'The Beta%'").fetchone()["id"]
    con.execute("UPDATE eval_queries SET expected_document_id = ?", (beta,))
    con.commit()
    report = run_eval(con, k=5, settings=small_corpus)
    assert report.n_queries == 1
    assert report.faithfulness_ratio == 1.0
