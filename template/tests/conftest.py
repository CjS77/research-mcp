"""Shared fixtures: an isolated settings/DB and a tiny synthetic corpus."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_kb.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Hermetic settings pointing all paths into a tmp dir, hashing embedder."""
    ref = tmp_path / "ref"
    docs = tmp_path / "docs"
    ref.mkdir()
    docs.mkdir()
    return Settings(
        db_path=tmp_path / "kb.sqlite",
        distilled_dir=tmp_path / "distilled",
        base_dir=tmp_path,
        reference_dir=ref,
        docs_dir=docs,
        embed_backend="hashing",
        embed_dim=256,
    )


@pytest.fixture
def small_corpus(settings: Settings) -> Settings:
    """Two cross-citing markdown sources (under the reference/ corpus root) so citation resolution links."""
    (settings.reference_dir / "alpha-forecasting-method.md").write_text(
        "# Alpha Method for Time-Series Forecasting\n\n"
        "## 1 Introduction\n\n"
        "We study forecasting methods where a model predicts future values from past observations. "
        "The method is accurate across a wide range of inputs.\n\n"
        "## 2 Construction\n\n"
        "**Theorem 1.** The estimator converges to the true mean under mild regularity assumptions. "
        "This statement is intentionally long to exercise the atomic-unit rule so that it "
        "should never be split across chunks regardless of the token budget in force here.\n\n"
        "## References\n\n"
        "[1] Beta et al. The Beta Method for Signal Denoising. 2020.\n"
        "[2] Someone. An Unrelated Work on Databases. 2019.\n",
        encoding="utf-8",
    )
    (settings.reference_dir / "beta-denoising-method.md").write_text(
        "# The Beta Method for Signal Denoising\n\n"
        "## 1 Overview\n\n"
        "Signal denoising removes noise from a measured series with no reference template.\n",
        encoding="utf-8",
    )
    return settings
