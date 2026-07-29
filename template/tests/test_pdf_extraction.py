"""Deterministic PDF extraction against the real reference papers (skipped if absent)."""

from __future__ import annotations

import pytest

from research_kb.corpus import _pdf_title
from research_kb.extract import extract_document
from tests.helpers import REFERENCE_DIR

SAMPLE = REFERENCE_DIR / "sample-paper.pdf"
pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="reference PDFs not present")


def test_pdf_extraction_has_structure_and_provenance():
    doc = extract_document(SAMPLE, "reference/sample-paper.pdf")
    assert doc.extractor == "pymupdf"
    assert doc.page_count and doc.page_count > 10
    assert doc.word_count > 1000

    headings = [ln for ln in doc.text.splitlines() if ln.startswith("## ")]
    assert any("Introduction" in h for h in headings)

    # page map is monotonic and covers the text
    assert doc.page_spans
    assert doc.page_at(0) == 1
    assert doc.page_at(len(doc.text) - 1) >= doc.page_at(0)


def test_pdf_title_reconstructed_from_largest_font():
    title = _pdf_title(SAMPLE, "sample-paper")
    assert title and " " in title
