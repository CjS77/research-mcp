"""LLM extraction via the Claude Code CLI (``claude -p``) — subscription auth, no API billing.

The Anthropic API path in :mod:`research_kb.extract.llm` meters every token. When a Claude subscription
is available, headless ``claude -p`` performs the same native-PDF transcription under the
subscription at no metered cost: it Reads the PDF and returns Markdown.

A full document overflows a single response's output ceiling — the observed failure mode is a
*silent* truncation (a long transcription stops mid-appendix). So the PDF is split into small
page-range segments, each transcribed in its own ``claude -p`` call, and the parts are concatenated
in document order into the ``llm.md`` artifact. This is the automated form of the manual
"split into parts" workaround, and it is bounded regardless of document length.

Long, dense segments also trip Claude's *output content filter* (a ``400 Output blocked by content
filtering policy`` on ~8 pages of dense prose). This is length-driven, not content-specific: the
identical pages transcribe cleanly when split smaller. So a filtered segment is bisected and each
half retried (:func:`_transcribe_range`); only a single page that still will not pass is left as a
clearly-marked gap, rather than losing the whole document.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pymupdf

from ..config import Settings, get_settings

_SYSTEM_PROMPT = (
    "You are a faithful PDF transcriber. Transcribe exactly what the document says; never summarize, "
    "correct, reword, or add commentary. Output only the transcription."
)

_SEGMENT_PROMPT = (
    "The file {pdf} contains pages {first}-{last} of an academic paper. Read it and transcribe those "
    "pages into faithful GitHub-flavored Markdown. Preserve every equation as LaTeX ($...$ inline, "
    "$$...$$ display), keep section numbers and titles, and keep tables. Do not summarize, correct, or "
    "reword. Do not add any preamble, explanation, or code fences — output only the Markdown "
    "transcription of these pages."
)


def claude_available(settings: Settings | None = None) -> bool:
    """True when the ``claude`` CLI is on PATH (the ``claude_cli`` backend can run)."""
    settings = settings or get_settings()
    return shutil.which(settings.claude_bin) is not None


def segment_ranges(page_count: int, seg_pages: int) -> list[tuple[int, int]]:
    """0-indexed inclusive ``(first, last)`` page ranges that tile the whole document."""
    seg_pages = max(1, seg_pages)
    return [(i, min(i + seg_pages - 1, page_count - 1)) for i in range(0, page_count, seg_pages)]


def build_command(pdf: Path, first: int, last: int, workdir: Path, settings: Settings) -> list[str]:
    """The ``claude -p`` argv for transcribing 1-indexed pages ``first..last`` of ``pdf``.

    Restricted to the Read tool, subscription auth (no ``--bare``, which would force API-key auth),
    permissions bypassed for the read-only session, and session persistence off for a clean batch.
    """
    return [
        settings.claude_bin,
        "-p",
        _SEGMENT_PROMPT.format(pdf=pdf.name, first=first, last=last),
        "--model", settings.claude_model,
        "--effort", settings.claude_effort,
        "--append-system-prompt", _SYSTEM_PROMPT,
        "--tools", "Read",
        "--permission-mode", "bypassPermissions",
        "--add-dir", str(workdir),
        "--no-session-persistence",
    ]


def _write_segment_pdf(src: pymupdf.Document, first: int, last: int, dest: Path) -> None:
    """Write a standalone PDF holding only pages ``first..last`` (0-indexed inclusive) of ``src``."""
    out = pymupdf.open()
    out.insert_pdf(src, from_page=first, to_page=last)
    out.save(dest)
    out.close()


# Signature of the output content-filter block (a 400 the CLI surfaces on stdout). Deterministic and
# length-driven, so it is answered by bisecting the segment rather than by retrying.
_FILTER_SIGNATURE = "content filtering policy"

# One transcription attempt resolves to exactly one of these outcomes.
_OK, _FILTERED, _ERROR = "ok", "filtered", "error"


def _run_claude(pdf: Path, first: int, last: int, workdir: Path, settings: Settings) -> tuple[str, str | None]:
    """Run ``claude -p`` once on a segment PDF. Returns ``(outcome, text)``.

    ``outcome`` is ``ok`` (text is the transcription), ``filtered`` (output content-filter block —
    deterministic, answered by splitting) or ``error`` (timeout / nonzero exit — transient, retried).
    """
    cmd = build_command(pdf, first, last, workdir, settings)
    try:
        result = subprocess.run(
            cmd, cwd=str(workdir), capture_output=True, text=True, timeout=settings.claude_timeout_s
        )
    except (subprocess.TimeoutExpired, OSError):
        return _ERROR, None
    text = result.stdout.strip()
    if result.returncode == 0 and text:
        return _OK, text
    combined = f"{result.stdout} {result.stderr}".lower()
    return (_FILTERED, None) if _FILTER_SIGNATURE in combined else (_ERROR, None)


def run_claude_extraction(path: Path, settings: Settings | None = None) -> str | None:
    """Transcribe a PDF to Markdown via segmented ``claude -p`` calls.

    A hard error on any segment after retries aborts the whole extraction (a partial transcription
    would manufacture false "present in one extractor only" divergences), returning None so the
    pipeline degrades to deterministic-only. Content-filter blocks are handled by bisection, not abort.
    """
    settings = settings or get_settings()
    if not claude_available(settings):
        return None
    try:
        src = pymupdf.open(path)
    except Exception:
        return None
    try:
        ranges = segment_ranges(src.page_count, settings.llm_segment_pages)
        parts: list[str] = []
        with tempfile.TemporaryDirectory() as td:
            workdir = Path(td)
            for first, last in ranges:
                part = _transcribe_range(src, first, last, workdir, path.stem, settings)
                if part is None:  # abort early — a partial transcription is worse than none
                    return None
                parts.append(part)
    finally:
        src.close()
    return "\n\n".join(parts) + "\n"


def _transcribe_range(
    src: pymupdf.Document, first: int, last: int, workdir: Path, stem: str, settings: Settings
) -> str | None:
    """Transcribe a 0-indexed inclusive page range, bisecting on a content-filter block.

    Transient errors are retried; a filter block bisects the range and retries each half (the same
    pages pass when smaller). A lone page that still will not pass becomes a marked gap so the rest of
    the paper survives; a hard error after retries aborts (returns None).
    """
    seg_pdf = workdir / f"{stem}_p{first + 1}-{last + 1}.pdf"
    _write_segment_pdf(src, first, last, seg_pdf)

    outcome, text = _ERROR, None
    for _attempt in range(max(1, settings.claude_max_retries + 1)):
        outcome, text = _run_claude(seg_pdf, first + 1, last + 1, workdir, settings)
        if outcome != _ERROR:  # ok and filtered are both deterministic — do not waste retries
            break

    if outcome == _OK:
        return f"<!-- pages {first + 1}-{last + 1} -->\n\n{text}"
    if outcome == _FILTERED:
        if last > first:
            mid = (first + last) // 2
            left = _transcribe_range(src, first, mid, workdir, stem, settings)
            right = _transcribe_range(src, mid + 1, last, workdir, stem, settings)
            return None if left is None or right is None else f"{left}\n\n{right}"
        return f"<!-- pages {first + 1}-{last + 1}: transcription unavailable (blocked by content filter) -->"
    return None
