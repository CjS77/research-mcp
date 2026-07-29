"""Tests for the ``claude -p`` LLM-extraction backend (subprocess mocked — no real CLI calls)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest

from research_kb.config import Settings
from research_kb.extract import claude_cli


@pytest.fixture(autouse=True)
def _claude_on_path(monkeypatch):
    """Pretend the ``claude`` CLI is installed so the mocked subprocess is actually reached.

    Every extraction test stubs ``subprocess.run`` but not the PATH probe. On a runner without the
    ``claude`` binary, ``claude_available`` short-circuits ``run_claude_extraction`` to ``None``
    before the mock runs, so the tests would only pass on a machine that happens to have Claude Code
    installed. Forcing availability makes them environment-independent.
    """
    monkeypatch.setattr(claude_cli, "claude_available", lambda settings=None: True)


def _blank_pdf(path, pages: int) -> None:
    doc = pymupdf.open()
    for _ in range(pages):
        doc.new_page()
    doc.save(path)
    doc.close()


def test_segment_ranges_tiles_the_document():
    assert claude_cli.segment_ranges(8, 8) == [(0, 7)]
    assert claude_cli.segment_ranges(52, 8) == [(0, 7), (8, 15), (16, 23), (24, 31), (32, 39), (40, 47), (48, 51)]
    assert claude_cli.segment_ranges(1, 8) == [(0, 0)]
    assert claude_cli.segment_ranges(10, 3) == [(0, 2), (3, 5), (6, 8), (9, 9)]


def test_build_command_uses_subscription_flags():
    s = Settings(claude_model="sonnet", claude_effort="high")
    cmd = claude_cli.build_command(Path("seg.pdf"), 1, 8, Path("/tmp/x"), s)
    assert cmd[:3] == ["claude", "-p", cmd[2]]
    assert "seg.pdf" in cmd[2] and "1-8" in cmd[2]
    assert "--model" in cmd and "sonnet" in cmd
    assert "--effort" in cmd and "high" in cmd
    assert cmd[cmd.index("--tools") + 1] == "Read"
    assert cmd[cmd.index("--permission-mode") + 1] == "bypassPermissions"
    # `--bare` would force API-key auth and defeat subscription billing avoidance.
    assert "--bare" not in cmd


def test_run_extraction_concatenates_segments(tmp_path, monkeypatch):
    pdf = tmp_path / "paper.pdf"
    _blank_pdf(pdf, 3)
    s = Settings(llm_segment_pages=2)

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout=f"SEG-{len(calls)}\n", stderr="")

    monkeypatch.setattr(claude_cli.subprocess, "run", fake_run)
    out = claude_cli.run_claude_extraction(pdf, s)

    assert len(calls) == 2  # 3 pages, 2 per segment -> ranges (0,1) and (2,2)
    assert "<!-- pages 1-2 -->" in out and "<!-- pages 3-3 -->" in out
    assert "SEG-1" in out and "SEG-2" in out


def test_failed_segment_aborts_whole_extraction(tmp_path, monkeypatch):
    pdf = tmp_path / "paper.pdf"
    _blank_pdf(pdf, 3)
    s = Settings(llm_segment_pages=2, claude_max_retries=1)

    attempts = {"n": 0}

    def fake_run(cmd, **kwargs):
        attempts["n"] += 1
        return SimpleNamespace(returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(claude_cli.subprocess, "run", fake_run)
    out = claude_cli.run_claude_extraction(pdf, s)

    assert out is None
    # first segment retried (max_retries + 1 = 2) then aborts before touching the second.
    assert attempts["n"] == 2


def test_timeout_is_swallowed_as_failure(tmp_path, monkeypatch):
    pdf = tmp_path / "paper.pdf"
    _blank_pdf(pdf, 1)
    s = Settings(llm_segment_pages=8, claude_max_retries=0)

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(claude_cli.subprocess, "run", fake_run)
    assert claude_cli.run_claude_extraction(pdf, s) is None


def test_generate_artifact_writes_llm_md(tmp_path, monkeypatch):
    from research_kb.extract import llm

    pdf = tmp_path / "paper.pdf"
    _blank_pdf(pdf, 2)
    distilled = tmp_path / "distilled"
    s = Settings(distilled_dir=distilled, llm_segment_pages=8, llm_extract_backend="claude_cli")

    monkeypatch.setattr(
        claude_cli.subprocess, "run",
        lambda cmd, **kw: SimpleNamespace(returncode=0, stdout="BODY", stderr=""),
    )
    path = llm.generate_llm_artifact(pdf, "paper", s)
    assert path == distilled / "paper" / "llm.md"
    assert "BODY" in path.read_text(encoding="utf-8")

    # Second call without force is a no-op that returns the existing artifact.
    path.write_text("EDITED", encoding="utf-8")
    again = llm.generate_llm_artifact(pdf, "paper", s)
    assert again.read_text(encoding="utf-8") == "EDITED"


def _page_span(cmd: list[str]) -> tuple[int, int]:
    """Recover the (first, last) page span a fake_run was asked to transcribe, from the prompt."""
    import re

    m = re.search(r"pages (\d+)-(\d+)", cmd[2])
    return int(m.group(1)), int(m.group(2))


def test_filtered_segment_bisects_and_recovers(tmp_path, monkeypatch):
    # A multi-page segment trips the content filter; the same pages pass once bisected to singles.
    pdf = tmp_path / "paper.pdf"
    _blank_pdf(pdf, 2)
    s = Settings(llm_segment_pages=2, claude_max_retries=2)

    calls: list[tuple[int, int]] = []

    def fake_run(cmd, **kwargs):
        first, last = _page_span(cmd)
        calls.append((first, last))
        if last > first:  # any multi-page range is blocked
            return SimpleNamespace(returncode=1, stdout="API Error: 400 Output blocked by content filtering policy", stderr="")
        return SimpleNamespace(returncode=0, stdout=f"PAGE-{first}\n", stderr="")

    monkeypatch.setattr(claude_cli.subprocess, "run", fake_run)
    out = claude_cli.run_claude_extraction(pdf, s)

    assert out is not None
    assert "<!-- pages 1-1 -->" in out and "<!-- pages 2-2 -->" in out
    assert "PAGE-1" in out and "PAGE-2" in out
    # The blocked (1,2) call is NOT retried (filter blocks are deterministic): 1 blocked + 2 singles.
    assert calls == [(1, 2), (1, 1), (2, 2)]


def test_single_page_filter_becomes_marked_gap(tmp_path, monkeypatch):
    # A lone page that will not pass the filter leaves a marked gap, not a lost paper.
    pdf = tmp_path / "paper.pdf"
    _blank_pdf(pdf, 1)
    s = Settings(llm_segment_pages=1, claude_max_retries=2)

    def fake_run(cmd, **kwargs):
        return SimpleNamespace(returncode=1, stdout="Output blocked by content filtering policy", stderr="")

    monkeypatch.setattr(claude_cli.subprocess, "run", fake_run)
    out = claude_cli.run_claude_extraction(pdf, s)

    assert out is not None  # paper survives
    assert "transcription unavailable" in out and "pages 1-1" in out


@pytest.mark.parametrize("backend,expected", [("none", None)])
def test_none_backend_skips_extraction(tmp_path, backend, expected):
    from research_kb.extract import llm

    pdf = tmp_path / "paper.pdf"
    _blank_pdf(pdf, 1)
    s = Settings(llm_extract_backend=backend)
    assert llm.run_live_extraction(pdf, s) is expected
