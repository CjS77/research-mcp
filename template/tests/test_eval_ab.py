"""Eval A/B: compare two embedding models on one gold set, each rebuilt into its own index.

Uses the deterministic hashing backend at two dimensions so the run is offline and fast — no model
download — while still exercising the rebuild-per-model path (each arm gets its own DB/vector space).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from research_kb.config import Settings
from research_kb.eval import ABModelSpec, run_ab_eval

GOLD = """
- query: What convergence guarantee does the Alpha forecasting estimator provide?
  expected_source_path: ref/alpha-forecasting-method.md
- query: How does the Beta method remove noise without a reference template?
  expected_source_path: ref/beta-denoising-method.md
"""


@pytest.fixture
def gold_path(tmp_path: Path) -> Path:
    p = tmp_path / "gold.yaml"
    p.write_text(GOLD, encoding="utf-8")
    return p


def test_ab_eval_scores_two_models(small_corpus: Settings, gold_path: Path):
    spec_a = ABModelSpec(model="hashing-256", dim=256, backend="hashing", label="A")
    spec_b = ABModelSpec(model="hashing-128", dim=128, backend="hashing", label="B")

    report = run_ab_eval(small_corpus, gold_path, spec_a, spec_b, k=5)

    # Both arms produced a full eval over the same gold set.
    assert report.report_a.n_queries == 2
    assert report.report_b.n_queries == 2
    assert 0.0 <= report.report_a.recall_at_k <= 1.0
    assert 0.0 <= report.report_b.recall_at_k <= 1.0
    # Faithfulness holds regardless of the embedding model (verbatim chunks are hash-checked).
    assert report.report_a.faithfulness_ratio == 1.0
    assert report.report_b.faithfulness_ratio == 1.0

    d = report.as_dict()
    assert d["a"]["dim"] == 256
    assert d["b"]["dim"] == 128
    assert d["k"] == 5


def test_ab_eval_isolates_each_model_in_its_own_db(small_corpus: Settings, gold_path: Path, tmp_path: Path):
    # A supplied workdir keeps each arm's DB separate and does not touch the primary index path.
    workdir = tmp_path / "ab"
    spec_a = ABModelSpec(model="m-a", dim=256, backend="hashing", label="A")
    spec_b = ABModelSpec(model="m-b", dim=256, backend="hashing", label="B")

    run_ab_eval(small_corpus, gold_path, spec_a, spec_b, k=5, workdir=workdir)

    assert (workdir / "A" / "kb.sqlite").exists()
    assert (workdir / "B" / "kb.sqlite").exists()
    # The primary DB path was never created by the A/B run.
    assert not small_corpus.db_path.exists()
