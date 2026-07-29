"""Deterministic extraction — the source of truth.

Default backend is ``pymupdf`` (fast, reproducible). It reconstructs markdown headings from
numbered-section patterns, known section words, and font-size so the structural chunker has
section boundaries and ``section_number`` provenance. ``marker`` is an optional higher-fidelity
backend; when unavailable the code falls back to pymupdf rather than failing.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import fitz  # pymupdf

from .base import ExtractedDoc, PageSpan

# "3", "3.2", "A.1" style section numbers followed by a Title-cased heading.
_SECTION_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)\.?\s+([A-Z][^.]{2,80})$")
# A line that is *only* a section number — headings are sometimes split from their title.
_BARE_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?$")
_KNOWN_HEADINGS = frozenset(
    {
        "abstract", "introduction", "background", "preliminaries", "related work", "conclusion",
        "conclusions", "references", "acknowledgements", "acknowledgments", "appendix",
        "notation", "our contributions", "security analysis", "discussion",
    }
)

_LineRec = tuple[str, float, bool]  # (text, max_font_size, bold)


def extract_pdf(path: Path, source_path: str, backend: str = "pymupdf") -> ExtractedDoc:
    if backend == "marker":
        marker_doc = _try_marker(path, source_path)
        if marker_doc is not None:
            return marker_doc
    return _extract_pymupdf(path, source_path)


def _page_blocks(doc: fitz.Document) -> list[list[list[_LineRec]]]:
    """pages → text blocks → lines, each line reduced to (text, max size, bold)."""
    pages: list[list[list[_LineRec]]] = []
    for page in doc:
        blocks: list[list[_LineRec]] = []
        data = page.get_text("dict")
        for block in data.get("blocks", []):
            if block.get("type", 0) != 0:  # skip image blocks
                continue
            lines: list[_LineRec] = []
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = "".join(s.get("text", "") for s in spans)
                if not text.strip():
                    continue
                size = max((s.get("size", 0.0) for s in spans), default=0.0)
                bold = any(int(s.get("flags", 0)) & 16 for s in spans)
                lines.append((text, size, bold))
            if lines:
                blocks.append(lines)
        pages.append(blocks)
    return pages


def _body_size(pages: list[list[list[_LineRec]]]) -> float:
    """Most common font size among paragraph-length lines (the body text size)."""
    sizes: Counter[float] = Counter()
    for blocks in pages:
        for block in blocks:
            for text, size, _bold in block:
                if len(text.strip()) >= 40:
                    sizes[round(size * 2) / 2] += 1
    if sizes:
        return sizes.most_common(1)[0][0]
    all_sizes = sorted(size for blocks in pages for block in blocks for _t, size, _b in block)
    return all_sizes[len(all_sizes) // 2] if all_sizes else 10.0


_STOPWORDS = frozenset(
    {"a", "an", "the", "of", "for", "and", "or", "to", "in", "on", "with", "without",
     "versus", "vs", "under", "from", "by", "as", "at", "via", "using"}
)


def _is_heading_title(title: str) -> bool:
    """Headings are short noun phrases, not sentences — filters out enumerated protocol steps."""
    return len(title) <= 72 and len(title.split()) <= 8


def _looks_like_heading_title(title: str) -> bool:
    """True when most significant words are capitalized — a noun-phrase title, not an imperative step."""
    tokens = re.findall(r"[A-Za-z][A-Za-z'’]*", title)
    significant = [t for t in tokens if t.lower() not in _STOPWORDS]
    if not significant:
        return False
    capitalized = sum(1 for t in significant if t[0].isupper())
    return capitalized / len(significant) >= 0.7


def _heading_level(text: str, size: float, body: float, page_index: int, title_seen: bool) -> int:
    """Return a markdown heading level (1-4) for a heading line, or 0 if it is body text."""
    lowered = text.lower().rstrip(":")
    m = _SECTION_NUM_RE.match(text)
    # Accept a numbered heading if it is visually larger than body OR reads like a title (not a step).
    if (
        m
        and not text.endswith((".", ":"))
        and _is_heading_title(m.group(2))
        and (size >= body + 0.5 or _looks_like_heading_title(m.group(2)))
    ):
        return min(2 + m.group(1).count("."), 4)
    if lowered in _KNOWN_HEADINGS and len(text) <= 40:
        return 2
    if size >= body + 1.0 and not text.endswith((".", ",", ";")) and _is_heading_title(text):
        if page_index == 0 and not title_seen and size >= body + 3.0:
            return 1  # paper title
        return 2 if size >= body + 2.0 else 3
    return 0


def _join_paragraph(lines: list[str]) -> str:
    """De-hyphenate line breaks and join wrapped lines into a single paragraph."""
    out = ""
    for ln in lines:
        ln = ln.strip()
        if not out:
            out = ln
        elif out.endswith("-") and len(out) >= 2 and out[-2].isalpha():
            out = out[:-1] + ln
        else:
            out = f"{out} {ln}"
    return out


def _extract_pymupdf(path: Path, source_path: str) -> ExtractedDoc:
    with fitz.open(path) as doc:
        pages = _page_blocks(doc)
        page_count = doc.page_count

    body = _body_size(pages)
    parts: list[str] = []
    page_spans: list[PageSpan] = []
    offset = 0
    title_seen = False

    pending_num: tuple[str, int] | None = None  # a bare section number awaiting its title

    def emit(s: str) -> None:
        nonlocal offset
        parts.append(s)
        offset += len(s)

    def flush_pending() -> None:
        nonlocal pending_num
        if pending_num:
            num, lvl = pending_num
            emit("#" * lvl + " " + num + "\n\n")
            pending_num = None

    for page_index, blocks in enumerate(pages):
        page_start = offset
        for block in blocks:
            para: list[str] = []
            for text, size, _bold in block:
                stripped = text.strip()
                if not stripped:
                    continue
                bare = _BARE_NUM_RE.match(stripped)
                if bare and size >= body + 1.0:
                    # A lone large-font number — buffer it to merge with the following title line.
                    if para:
                        emit(_join_paragraph(para) + "\n\n")
                        para = []
                    flush_pending()
                    num = bare.group(1)
                    pending_num = (num, min(2 + num.count("."), 4))
                    continue
                level = _heading_level(stripped, size, body, page_index, title_seen)
                if level:
                    if para:
                        emit(_join_paragraph(para) + "\n\n")
                        para = []
                    if pending_num:
                        num, lvl = pending_num
                        emit("#" * lvl + " " + num + " " + stripped + "\n\n")
                        pending_num = None
                    else:
                        if level == 1:
                            title_seen = True
                        emit("#" * level + " " + stripped + "\n\n")
                else:
                    flush_pending()  # bare number not followed by a title → emit it alone
                    para.append(stripped)
            if para:
                emit(_join_paragraph(para) + "\n\n")
        flush_pending()
        if offset > page_start:
            page_spans.append(PageSpan(start=page_start, end=offset, page=page_index + 1))

    return ExtractedDoc(
        source_path=source_path,
        text="".join(parts).strip() + "\n",
        page_spans=page_spans,
        page_count=page_count,
        extractor="pymupdf",
    )


def _try_marker(path: Path, source_path: str) -> ExtractedDoc | None:
    """Best-effort marker-pdf extraction; returns None so the caller falls back to pymupdf.

    marker's output does not expose a char→page map in a stable way across versions, so page
    provenance is coarser here; enable only when layout fidelity matters more than page refs.
    """
    try:
        from marker.converters.pdf import PdfConverter  # type: ignore
        from marker.models import create_model_dict  # type: ignore
        from marker.output import text_from_rendered  # type: ignore
    except Exception:
        return None
    try:
        converter = PdfConverter(artifact_dict=create_model_dict())
        rendered = converter(str(path))
        text, _meta, _images = text_from_rendered(rendered)
        return ExtractedDoc(source_path=source_path, text=text, page_spans=[], page_count=None, extractor="marker")
    except Exception:
        return None
