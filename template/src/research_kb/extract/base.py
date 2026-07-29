"""Shared extraction types: the transcribed document plus its char-offset → page map."""

from __future__ import annotations

import bisect
import hashlib

from pydantic import BaseModel, Field


class PageSpan(BaseModel):
    """A contiguous character range in the extracted text belonging to one source page."""

    start: int  # char offset, inclusive
    end: int    # char offset, exclusive
    page: int   # 1-based page number


class ExtractedDoc(BaseModel):
    """Deterministic transcription of a source file: text + provenance.

    ``text`` is markdown (headings reconstructed for PDFs). ``page_spans`` lets any downstream
    char range be mapped back to a source page; empty for page-less markdown sources.
    """

    source_path: str
    text: str
    page_spans: list[PageSpan] = Field(default_factory=list)
    page_count: int | None = None
    extractor: str = "unknown"

    def model_post_init(self, _ctx: object) -> None:
        # Cache span starts for fast page lookup.
        self._starts = [s.start for s in self.page_spans]

    def page_at(self, char_offset: int) -> int | None:
        """Page number containing ``char_offset`` (None for page-less sources)."""
        if not self.page_spans:
            return None
        idx = bisect.bisect_right(self._starts, char_offset) - 1
        idx = max(0, min(idx, len(self.page_spans) - 1))
        return self.page_spans[idx].page

    def page_range(self, start: int, end: int) -> tuple[int | None, int | None]:
        """(first_page, last_page) spanned by a char range."""
        return self.page_at(start), self.page_at(max(start, end - 1))

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def span_hash(text: str) -> str:
    """Stable hash of a verbatim span, for faithfulness auditing."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
