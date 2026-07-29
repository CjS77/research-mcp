"""Indexing orchestration: run each document through the pipeline into SQLite.

Per document: deterministic extract → (optional LLM extract + divergence cross-check) → enrichment →
chunk → embed → store, with distillation artifacts committed to ``distilled/<stem>/`` and progress
tracked in ``indexing_jobs``. Incremental: an unchanged ``content_hash`` skips the document.
The core-tier validation gate blocks a document whose cross-check has unresolved semantic
divergences unless it has been human-validated.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .chunk import build_derived_chunk, chunk_document
from .citations import resolve_citations, store_citations
from .config import Settings, get_settings
from .corpus import CorpusItem, scan
from .db import init_db
from .divergence import detect_divergences, render_report
from .embed import get_embedder
from .embed.base import EmbeddingProvider
from .enrich import EnrichmentResult, enrich
from .extract import extract_document, parse_llm_markdown
from .extract.base import ExtractedDoc
from .extract.llm import get_llm_markdown
from .models import Document
from .store import delete_document_data, get_document_by_path, insert_chunk, set_chunk_embedded, upsert_document


@dataclass
class IndexSummary:
    indexed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)
    pruned: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    chunks_created: int = 0
    citations_resolved: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "indexed": self.indexed, "skipped": self.skipped, "blocked": self.blocked,
            "pruned": self.pruned, "failed": self.failed, "chunks_created": self.chunks_created,
            "citations_resolved": self.citations_resolved,
        }


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_artifact(settings: Settings, stem: str, name: str, content: str) -> None:
    out_dir = settings.artifact_dir(stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_text(content, encoding="utf-8")


def _render_enriched(item: CorpusItem, extracted: ExtractedDoc, enr: EnrichmentResult) -> str:
    """Derived ``enriched.md``: frontmatter + section outline. Verbatim spans live in verbatim.md."""
    fm = [
        "---",
        f"title: {item.title}",
        f"source_path: {item.source_path}",
        f"doc_type: {item.doc_type}",
        f"tier: {item.tier}",
        f"year: {enr.year if enr.year is not None else ''}",
        f"facet_a: {enr.facet_a}",
        f"facet_b: {enr.facet_b}",
        f"indexed_extractor: {extracted.extractor}",
        f"enrichment_source: {enr.source}",
        "content_kind: derived",
        "---",
        "",
        f"> {enr.notation_note}" if enr.notation_note else "",
        "",
        "## Section outline",
        "",
    ]
    fm += [f"- **{(s.section_number or '').strip()} {s.section_title}** — {s.summary}" for s in enr.section_summaries]
    return "\n".join(fm) + "\n"


def _start_job(con: sqlite3.Connection, source_path: str) -> int:
    cur = con.execute(
        "INSERT INTO indexing_jobs (source_path, status, started_at) VALUES (?, 'processing', ?)",
        (source_path, _now()),
    )
    rowid = cur.lastrowid
    assert rowid is not None  # set by the INSERT above
    return rowid


def _finish_job(con: sqlite3.Connection, job_id: int, status: str, chunks: int = 0, error: str | None = None) -> None:
    con.execute(
        "UPDATE indexing_jobs SET status = ?, completed_at = ?, chunks_created = ?, error_message = ? WHERE id = ?",
        (status, _now(), chunks, error, job_id),
    )


def index_item(
    con: sqlite3.Connection,
    item: CorpusItem,
    embedder: EmbeddingProvider | None,
    settings: Settings,
    force: bool = False,
) -> str:
    """Index one document. Returns 'indexed' | 'skipped' | 'blocked' | 'failed'."""
    existing = get_document_by_path(con, item.source_path)
    if existing and existing.content_hash == item.content_hash and not force:
        return "skipped"

    job_id = _start_job(con, item.source_path)
    try:
        stem = item.abs_path.stem
        # Deterministic extraction is always the guardrail: it is written to verbatim.md and is the
        # baseline for the cross-check. When a trustworthy LLM transcription exists it becomes what we
        # actually chunk and index (the faithful math/layout extractor), but verbatim.md is retained.
        deterministic = extract_document(item.abs_path, item.source_path, settings)
        _write_artifact(settings, stem, "verbatim.md", deterministic.text)

        blocking = False
        llm_md = get_llm_markdown(item.abs_path, stem, settings)
        if llm_md:
            report = detect_divergences(item.source_path, deterministic.text, llm_md)
            _write_artifact(settings, stem, "divergence-report.md", render_report(report))
            blocking = report.has_blocking

        # Index the LLM extraction only once it is trustworthy: cross-check clean, or human-validated.
        validated = bool(existing.validated) if existing else False
        use_llm = bool(llm_md) and (not blocking or validated)
        indexed = parse_llm_markdown(item.source_path, llm_md) if use_llm and llm_md is not None else deterministic

        enr = enrich(indexed, item.title, settings)
        _write_artifact(settings, stem, "enriched.md", _render_enriched(item, indexed, enr))

        doc = Document(
            source_path=item.source_path, doc_type=item.doc_type, tier=item.tier, title=item.title,
            phase=item.phase, facet_b=enr.facet_b, facet_a=enr.facet_a, year=enr.year,
            page_count=deterministic.page_count, validated=validated,
            content_hash=item.content_hash, word_count=indexed.word_count,
        )
        doc_id = upsert_document(con, doc)
        delete_document_data(con, doc_id)

        # Core-tier validation gate: a blocking semantic divergence stops indexing until validated.
        if item.tier == "core" and blocking and not doc.validated:
            _finish_job(con, job_id, "blocked", 0, "unresolved semantic divergence; run `research-kb validate`")
            return "blocked"

        chunks = _store_chunks(con, doc_id, doc.title, indexed, enr, embedder, settings)
        store_citations(con, doc_id, indexed)
        _finish_job(con, job_id, "completed", chunks)
        return "indexed"
    except Exception as exc:  # keep the batch going; the job row records the failure
        _finish_job(con, job_id, "failed", 0, f"{type(exc).__name__}: {exc}")
        raise


def _store_chunks(
    con: sqlite3.Connection,
    doc_id: int,
    title: str,
    extracted: ExtractedDoc,
    enr: EnrichmentResult,
    embedder: EmbeddingProvider | None,
    settings: Settings,
) -> int:
    tree = chunk_document(extracted, doc_id, title, enr, settings)
    embed_targets: list[tuple[int, str]] = []
    max_index = -1
    for pb in tree:
        parent_id = insert_chunk(con, pb.parent)
        max_index = max(max_index, pb.parent.chunk_index)
        for child in pb.children:
            child.parent_chunk_id = parent_id
            cid = insert_chunk(con, child)
            max_index = max(max_index, child.chunk_index)
            if child.embedded:
                embed_targets.append((cid, child.embed_input or child.content))

    derived = build_derived_chunk(doc_id, title, enr, max_index + 1)
    if derived is not None:
        did = insert_chunk(con, derived)
        embed_targets.append((did, derived.embed_input or derived.content))

    if embedder is not None and embed_targets:
        vectors = embedder.embed([text for _, text in embed_targets])
        for (cid, _), vector in zip(embed_targets, vectors, strict=True):
            set_chunk_embedded(con, cid, vector)

    total_chunks = sum(1 + len(pb.children) for pb in tree) + (1 if derived else 0)
    return total_chunks


def index_corpus(
    settings: Settings | None = None,
    roots: list[Path] | None = None,
    force: bool = False,
    embed: bool = True,
) -> IndexSummary:
    """Index (or refresh) the whole corpus, then resolve citation edges across it."""
    settings = settings or get_settings()
    con = init_db(settings)
    embedder = get_embedder(settings) if embed else None

    summary = IndexSummary()
    for item in scan(settings, roots):
        try:
            outcome = index_item(con, item, embedder, settings, force)
        except Exception as exc:  # noqa: BLE001 — one bad doc shouldn't abort the batch
            summary.failed.append((item.source_path, f"{type(exc).__name__}: {exc}"))
            con.commit()
            continue
        # outcome is one of the list-valued summary fields: indexed | skipped | blocked
        getattr(summary, outcome).append(item.source_path)
        con.commit()

    summary.pruned = _prune_missing(con, settings)
    con.commit()

    summary.citations_resolved = resolve_citations(con)
    con.commit()

    row = con.execute("SELECT COUNT(*) FROM chunks").fetchone()
    summary.chunks_created = int(row[0])
    return summary


def _prune_missing(con: sqlite3.Connection, settings: Settings) -> list[str]:
    """Drop documents (and their chunks/citations) whose source file no longer exists."""
    pruned: list[str] = []
    for row in con.execute("SELECT id, source_path FROM documents").fetchall():
        if not (settings.base_dir / row["source_path"]).exists():
            delete_document_data(con, row["id"])
            con.execute("DELETE FROM documents WHERE id = ?", (row["id"],))
            pruned.append(row["source_path"])
    return pruned
