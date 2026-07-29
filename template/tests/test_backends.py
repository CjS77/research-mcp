"""Distill backend registry: dispatch, index-time safety, and reserved names."""

from __future__ import annotations

from research_kb.config import Settings
from research_kb.extract import backends, llm


def test_shipped_backends_registered():
    assert {"claude_cli", "api", "none"} <= set(backends.backend_names())


def test_index_time_safety_flags():
    assert backends.get_backend("api").index_time_safe is True
    assert backends.get_backend("none").index_time_safe is True
    # the heavy CLI backend must not be index-time-safe (index would fan out subprocesses)
    assert backends.get_backend("claude_cli").index_time_safe is False


def test_reserved_names_not_yet_registered():
    assert backends.get_backend("codex_cli") is None
    assert backends.get_backend("opencode_cli") is None


def test_run_live_extraction_none_and_unknown(tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.7")
    assert llm.run_live_extraction(pdf, Settings(llm_extract_backend="none")) is None
    assert llm.run_live_extraction(pdf, Settings(llm_extract_backend="bogus")) is None


def test_get_llm_markdown_never_spawns_heavy_backend(tmp_path):
    # claude_cli is not index_time_safe, so index-time extraction returns None (must run `distill`).
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.7")
    s = Settings(llm_extract_backend="claude_cli", distilled_dir=tmp_path / "distilled")
    assert llm.get_llm_markdown(pdf, "x", s) is None
