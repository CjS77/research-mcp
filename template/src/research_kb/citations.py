"""Citation graph: edges extracted at index time, resolved against the corpus.

Resolved edges (cited work is in-corpus) power ``kb_follow_citations``. Unresolved edges are a
ranked acquisition list — the "cited-but-missing" works to grow the KB.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from .extract.base import ExtractedDoc
from .models import Citation
from .store import insert_citation, list_documents

_REF_HEADING_RE = re.compile(r"^#{1,6}\s+(?:\d+\.?\s+)?(references|bibliography)\s*$", re.IGNORECASE | re.MULTILINE)
_BRACKET_MARKER_RE = re.compile(r"\[\d{1,3}\]")
_NUM_LINE_RE = re.compile(r"^\s*\d{1,3}\.\s+", re.MULTILINE)
_STOPWORDS = frozenset(
    {"the", "and", "for", "with", "from", "using", "under", "over", "into", "based", "via", "a",
     "an", "of", "on", "in", "to", "by", "at", "is", "are", "new", "toward", "towards"}
)


def _find_references_offset(text: str) -> int | None:
    matches = list(_REF_HEADING_RE.finditer(text))
    return matches[-1].end() if matches else None


def _split_references(block: str) -> list[str]:
    """Split a references block into individual entries by the dominant marker style."""
    markers = list(_BRACKET_MARKER_RE.finditer(block))
    if len(markers) >= 3:
        entries = []
        for i, m in enumerate(markers):
            end = markers[i + 1].start() if i + 1 < len(markers) else len(block)
            entries.append(block[m.end():end])
        return entries
    if len(_NUM_LINE_RE.findall(block)) >= 3:
        parts = _NUM_LINE_RE.split(block)
        return parts[1:] if parts else []
    return re.split(r"\n\s*\n", block)


def _clean_ref(entry: str) -> str:
    return re.sub(r"\s+", " ", entry).strip(" .\n\t")


def extract_citations(extracted: ExtractedDoc) -> list[tuple[str, int | None]]:
    """Return (raw_reference, page) pairs from the document's reference section."""
    offset = _find_references_offset(extracted.text)
    if offset is None:
        return []
    block = extracted.text[offset:]
    out: list[tuple[str, int | None]] = []
    cursor = offset
    for entry in _split_references(block):
        ref = _clean_ref(entry)
        if 20 <= len(ref) <= 600:
            page = extracted.page_at(cursor)
            out.append((ref, page))
        cursor += len(entry)
    return out


def store_citations(con: sqlite3.Connection, from_document_id: int, extracted: ExtractedDoc) -> int:
    refs = extract_citations(extracted)
    for ref, page in refs:
        insert_citation(
            con,
            Citation(from_document_id=from_document_id, to_reference=ref, context_snippet=ref[:160], page_ref=page),
        )
    return len(refs)


def _title_tokens(title: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", title.lower()) if t not in _STOPWORDS and len(t) > 2]


def _is_subsequence(needle: list[str], hay_words: list[str]) -> bool:
    """True if ``needle`` appears in ``hay_words`` in order (gaps allowed), matching whole words."""
    i = 0
    for w in hay_words:
        if i < len(needle) and w == needle[i]:
            i += 1
    return i == len(needle)


def _ngram_match(needle_tokens: list[str], hay_words: list[str], n: int = 3) -> bool:
    """Match if n consecutive significant title tokens appear, in order, among the reference words.

    Ordered-subsequence (not strict adjacency) so internal stopwords in the title don't defeat it.
    """
    if len(needle_tokens) < n:
        return len(needle_tokens) >= 2 and _is_subsequence(needle_tokens, hay_words)
    return any(
        _is_subsequence(needle_tokens[i:i + n], hay_words) for i in range(len(needle_tokens) - n + 1)
    )


def resolve_citations(con: sqlite3.Connection) -> int:
    """Resolve unresolved citation edges to in-corpus documents by title n-gram match. Returns count."""
    documents = list_documents(con)
    catalog = [(d.id, _title_tokens(d.title)) for d in documents if d.id and len(_title_tokens(d.title)) >= 2]
    rows = con.execute("SELECT id, from_document_id, to_reference FROM citations WHERE to_document_id IS NULL").fetchall()
    resolved = 0
    for row in rows:
        hay_words = re.sub(r"[^a-z0-9]+", " ", row["to_reference"].lower()).split()
        for doc_id, tokens in catalog:
            if doc_id == row["from_document_id"]:
                continue
            if _ngram_match(tokens, hay_words):
                con.execute("UPDATE citations SET to_document_id = ? WHERE id = ?", (doc_id, row["id"]))
                resolved += 1
                break
    return resolved


def acquisition_targets(con: sqlite3.Connection, limit: int = 20) -> list[dict[str, object]]:
    """Cited-but-missing works, ranked by how many corpus docs cite them."""
    rows = con.execute("SELECT to_reference FROM citations WHERE to_document_id IS NULL").fetchall()
    counts: dict[str, dict[str, Any]] = {}
    for row in rows:
        ref = row["to_reference"]
        key = re.sub(r"[^a-z0-9]+", " ", ref.lower())[:80].strip()
        if not key:
            continue
        slot = counts.setdefault(key, {"reference": ref[:200], "citations": 0})
        slot["citations"] = int(slot["citations"]) + 1
    ranked = sorted(counts.values(), key=lambda d: int(d["citations"]), reverse=True)
    return ranked[:limit]
