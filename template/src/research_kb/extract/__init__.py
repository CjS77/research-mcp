"""Extraction layer: deterministic (source of truth) + LLM (second opinion) transcription.

``extract_document`` dispatches on file type and configured backend, always returning an
:class:`~research_kb.extract.base.ExtractedDoc` with page provenance where the source has pages.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Settings, get_settings
from .base import ExtractedDoc
from .deterministic import extract_pdf
from .llm_doc import parse_llm_markdown
from .markdown_doc import extract_markdown


def extract_document(path: Path, source_path: str, settings: Settings | None = None) -> ExtractedDoc:
    """Deterministically extract a source file into text + page map."""
    settings = settings or get_settings()
    if path.suffix.lower() == ".pdf":
        return extract_pdf(path, source_path, backend=settings.pdf_extractor)
    return extract_markdown(path, source_path)


__all__ = ["ExtractedDoc", "extract_document", "extract_pdf", "extract_markdown", "parse_llm_markdown"]
