"""Extraction for markdown and plain-text sources — already text, no transcription needed."""

from __future__ import annotations

from pathlib import Path

from .base import ExtractedDoc


def extract_markdown(path: Path, source_path: str) -> ExtractedDoc:
    """Load a markdown/plain-text source verbatim. Page-less, so provenance is section-level, not page-level."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return ExtractedDoc(
        source_path=source_path,
        text=text,
        page_spans=[],
        page_count=None,
        extractor="text" if path.suffix.lower() == ".txt" else "markdown",
    )
