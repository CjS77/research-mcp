"""Pydantic domain models shared across extraction, indexing, search, and the MCP surface."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

DocType = Literal["paper", "research", "assessment", "sketch", "spec"]
Tier = Literal["core", "breadth"]
ContentKind = Literal["verbatim", "derived_summary", "enrichment"]
ChunkType = Literal["paragraph", "table", "code", "math", "theorem", "protocol", "heading"]

# Atomic units are never split even when oversized.
ATOMIC_CHUNK_TYPES: frozenset[str] = frozenset({"theorem", "protocol", "table", "code", "math"})


class Document(BaseModel):
    """A row of the ``documents`` table."""

    id: int | None = None
    source_path: str
    doc_type: DocType
    tier: Tier = "breadth"
    title: str
    phase: int | None = None
    # Named facets: {facet_name: [values]}. The declared axes live in the profile (config.FacetSpec);
    # a document carries only the facets it exhibits.
    facets: dict[str, list[str]] = Field(default_factory=dict)
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    page_count: int | None = None
    validated: bool = False
    validated_at: datetime | None = None
    content_hash: str | None = None
    word_count: int | None = None


class Chunk(BaseModel):
    """A provenance-carrying chunk. ``embedding`` lives in ``vec_chunks``, not here."""

    id: int | None = None
    document_id: int
    chunk_index: int
    content: str
    content_kind: ContentKind
    section_number: str | None = None
    section_title: str | None = None
    chunk_type: ChunkType
    page_start: int | None = None
    page_end: int | None = None
    verbatim_hash: str | None = None
    parent_chunk_id: int | None = None
    token_count: int | None = None
    embed_input: str | None = None
    embedded: bool = False


class Citation(BaseModel):
    """A citation edge. ``to_document_id`` is set when the cited work is in-corpus."""

    id: int | None = None
    from_document_id: int
    to_reference: str
    to_document_id: int | None = None
    context_snippet: str | None = None
    page_ref: int | None = None


class SearchHit(BaseModel):
    """A single retrieval result with full provenance (the ``kb_search`` shape)."""

    chunk_id: int
    document_id: int
    paper: str
    snippet: str
    section_number: str | None = None
    section_title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    content_kind: ContentKind
    chunk_type: ChunkType
    score: float
    retrieval: Literal["semantic", "keyword", "hybrid"] = "hybrid"


class EvalQuery(BaseModel):
    """A gold-set query with its expected provenance target."""

    id: int | None = None
    query: str
    expected_document_id: int | None = None
    expected_section: str | None = None
    expected_page: int | None = None
    notes: str | None = None
