"""Build an :class:`ExtractedDoc` from a committed ``llm.md`` transcription.

Once a document's LLM transcription has cleared the divergence cross-check (it is the *more* faithful
extractor for math and layout — the deterministic extractor mangles inline notation), it, not the
deterministic text, is what we chunk and index. The segmented ``claude -p`` output carries
``<!-- pages X-Y -->`` markers, so page provenance survives: each marker's body is attributed to its
page range (spread evenly across the range, since the exact intra-segment page breaks are unknown).
The deterministic ``verbatim.md`` is retained as the independent guardrail and audit trail.
"""

from __future__ import annotations

import re

from .base import ExtractedDoc, PageSpan

# `<!-- pages 12-15 -->` and the gap form `<!-- pages 12-15: transcription unavailable (...) -->`.
_PAGE_MARKER_RE = re.compile(r"[ \t]*<!--\s*pages\s+(\d+)-(\d+).*?-->[ \t]*\n?")


def _distribute_pages(spans: list[PageSpan], start: int, end: int, first: int, last: int) -> None:
    """Tile the char range ``[start, end)`` with one PageSpan per page in ``first..last`` (even split)."""
    n_pages = last - first + 1
    length = end - start
    if n_pages <= 1 or length <= 0:
        spans.append(PageSpan(start=start, end=end, page=first))
        return
    step = length / n_pages
    for k in range(n_pages):
        s = start + round(k * step)
        e = end if k == n_pages - 1 else start + round((k + 1) * step)
        spans.append(PageSpan(start=s, end=e, page=first + k))


def parse_llm_markdown(source_path: str, md: str, extractor: str = "claude_cli") -> ExtractedDoc:
    """Parse a segmented ``llm.md`` into text + page map, stripping the ``<!-- pages -->`` markers.

    With no markers (e.g. a hand-assembled transcription), the text is kept verbatim and provenance
    falls back to section level (empty ``page_spans``), exactly like a page-less markdown source.
    """
    markers = list(_PAGE_MARKER_RE.finditer(md))
    if not markers:
        return ExtractedDoc(source_path=source_path, text=md, page_spans=[], page_count=None, extractor=extractor)

    parts: list[str] = []
    spans: list[PageSpan] = []
    pos = 0

    preamble = md[: markers[0].start()]
    if preamble.strip():
        parts.append(preamble)
        pos += len(preamble)

    for i, m in enumerate(markers):
        first, last = int(m.group(1)), int(m.group(2))
        body = md[m.end() : (markers[i + 1].start() if i + 1 < len(markers) else len(md))]
        if not body.strip():  # a content-filter gap contributes no text and no page span
            continue
        start = pos
        parts.append(body)
        pos += len(body)
        _distribute_pages(spans, start, pos, first, last)

    page_count = max((int(m.group(2)) for m in markers), default=None)
    return ExtractedDoc(source_path=source_path, text="".join(parts), page_spans=spans, page_count=page_count, extractor=extractor)
