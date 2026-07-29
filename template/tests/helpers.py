"""Test helpers and constants shared across test modules."""

from __future__ import annotations

from pathlib import Path

from research_kb.extract.base import ExtractedDoc, PageSpan

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "reference"


def make_extracted(text: str, page_len: int = 200) -> ExtractedDoc:
    """Wrap text as an ExtractedDoc with synthetic page spans of ``page_len`` chars each."""
    spans = [
        PageSpan(start=i, end=min(i + page_len, len(text)), page=(i // page_len) + 1)
        for i in range(0, max(len(text), 1), page_len)
    ]
    return ExtractedDoc(source_path="synthetic.md", text=text, page_spans=spans, page_count=len(spans), extractor="test")
