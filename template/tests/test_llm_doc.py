"""Building an ExtractedDoc from a segmented ``llm.md`` (page markers -> page map)."""

from __future__ import annotations

from research_kb.extract.llm_doc import parse_llm_markdown

_SAMPLE = (
    "<!-- pages 1-2 -->\n\n# Method\n\nAbstract and introduction text here.\n\n"
    "<!-- pages 3-6 -->\n\n## 2 Preliminaries\n\n"
    + ("Body text for the preliminaries section spanning four source pages. " * 12)
    + "\n\n<!-- pages 7-7 -->\n\n## 3 Conclusion\n\nFinal remarks.\n"
)


def test_markers_are_stripped_from_indexed_text():
    doc = parse_llm_markdown("reference/x.pdf", _SAMPLE)
    assert "<!--" not in doc.text and "pages 1-2" not in doc.text
    assert doc.text.lstrip().startswith("# Method")
    assert doc.page_count == 7
    assert doc.extractor == "claude_cli"


def test_page_lookup_maps_offsets_to_source_pages():
    doc = parse_llm_markdown("reference/x.pdf", _SAMPLE)
    # First segment (pages 1-2): the very start is page 1.
    assert doc.page_at(0) == 1
    # Conclusion is on page 7.
    assert doc.page_at(doc.text.index("Final remarks")) == 7


def test_multi_page_segment_distributes_pages():
    doc = parse_llm_markdown("reference/x.pdf", _SAMPLE)
    prelim = doc.text.index("2 Preliminaries")
    body_end = doc.text.index("## 3 Conclusion")
    # A pages 3-6 segment should attribute its start near page 3 and its tail near page 6.
    assert doc.page_at(prelim) == 3
    assert doc.page_at(body_end - 5) == 6
    # Every page in the segment is represented.
    seen = {doc.page_at(o) for o in range(prelim, body_end, 20)}
    assert {3, 4, 5, 6} <= seen


def test_no_markers_falls_back_to_pageless():
    md = "# Hand-assembled\n\nNo page markers here.\n"
    doc = parse_llm_markdown("reference/y.pdf", md)
    assert doc.page_spans == []
    assert doc.page_count is None
    assert doc.page_at(0) is None
    assert doc.text == md


def test_content_filter_gap_contributes_no_span_but_keeps_flow():
    md = (
        "<!-- pages 1-1 -->\n\n# A\n\nPage one body.\n\n"
        "<!-- pages 2-2: transcription unavailable (blocked by content filter) -->\n\n"
        "<!-- pages 3-3 -->\n\n# C\n\nPage three body.\n"
    )
    doc = parse_llm_markdown("reference/z.pdf", md)
    assert "transcription unavailable" not in doc.text  # the gap marker is stripped too
    assert doc.page_at(doc.text.index("Page one body")) == 1
    assert doc.page_at(doc.text.index("Page three body")) == 3
    # Page 2 produced no content, so it simply has no span (no phantom text attributed to it).
    assert {s.page for s in doc.page_spans} == {1, 3}
