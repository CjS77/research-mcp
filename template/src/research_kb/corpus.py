"""Corpus discovery: scan ``reference/`` — the single corpus root — and infer metadata.

The corpus is exactly the external material under ``reference/`` (PDFs plus ``.md``/``.txt`` text
sources). ``docs/`` (the project's own research) and ``work/`` (server state) are never
scanned. Tier/type/phase are inferred from filename conventions (``<phase>.<seq>-<slug>.md``) and
the core-source set (see ``config``). ``content_hash`` drives incremental updates.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import fitz  # pymupdf
from pydantic import BaseModel

from .config import Settings, get_settings
from .models import DocType, Tier

# Subdirectory names under the scan root that are never corpus material (defensive; the corpus
# root should hold only external sources).
IGNORE_DIRS: frozenset[str] = frozenset({"kb", "_sweep"})

_PHASE_RE = re.compile(r"^(\d+)\.")
_H1_RE = re.compile(r"^#\s+(.*\S)\s*$", re.MULTILINE)


class CorpusItem(BaseModel):
    """A discovered source document with inferred metadata (pre-distillation)."""

    source_path: str  # posix, relative to repo root — the stable identity used in `documents`
    abs_path: Path
    doc_type: DocType
    tier: Tier
    title: str
    phase: int | None
    content_hash: str
    is_pdf: bool


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _clean_stem(stem: str) -> str:
    """Human-readable fallback title from a filename stem."""
    return re.sub(r"[-_]+", " ", re.sub(r"^\d+\.\d+-", "", stem)).strip().title()


_JUNK_TITLE_RE = re.compile(r"\.(tex|dvi|pdf|docx?|aux|log)\b", re.IGNORECASE)


def _title_from_first_page(path: Path) -> str:
    """Reconstruct the title from the largest-font text at the top of page 1."""
    try:
        with fitz.open(path) as doc:
            if doc.page_count == 0:
                return ""
            data = doc[0].get_text("dict")
    except Exception:
        return ""

    lines: list[tuple[float, str]] = []  # (max span size, text) in reading order
    for block in data.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(s.get("text", "") for s in spans).strip()
            if text:
                lines.append((max((s.get("size", 0.0) for s in spans), default=0.0), text))
    if not lines:
        return ""

    max_size = max(size for size, _ in lines)
    # Collect the run of top-of-page lines set at (near) the largest size — the title block.
    title_parts: list[str] = []
    for size, text in lines:
        if size >= max_size - 0.5:
            title_parts.append(text)
        elif title_parts:
            break
    title = re.sub(r"\s+", " ", " ".join(title_parts)).strip().rstrip("∗*†‡§¶ ")
    return title if 8 <= len(title) <= 160 and " " in title else ""


def _pdf_title(path: Path, stem: str) -> str:
    """Best available title: sane embedded metadata → largest-font page-1 text → cleaned stem."""
    try:
        with fitz.open(path) as doc:
            meta_title = ((doc.metadata or {}).get("title") or "").strip()
    except Exception:
        meta_title = ""
    if len(meta_title) >= 10 and " " in meta_title and not _JUNK_TITLE_RE.search(meta_title):
        return meta_title
    return _title_from_first_page(path) or _clean_stem(stem)


def _text_meta(path: Path) -> tuple[DocType, int | None, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    m = _H1_RE.search(text)
    title = m.group(1).strip() if m else _clean_stem(path.stem)

    name = path.stem.lower()
    if "assessment" in name:
        doc_type: DocType = "assessment"
    elif "sketch" in name:
        doc_type = "sketch"
    elif name.startswith("rfc") or "specification" in name or name == "design-specification":
        doc_type = "spec"
    else:
        doc_type = "research"

    phase_match = _PHASE_RE.match(path.stem)
    phase = int(phase_match.group(1)) if phase_match else None
    return doc_type, phase, title


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def scan(settings: Settings | None = None, roots: list[Path] | None = None) -> list[CorpusItem]:
    """Discover all indexable documents under the given roots (default: the reference/ corpus root)."""
    settings = settings or get_settings()
    base = settings.base_dir
    roots = roots or [settings.reference_dir]
    items: list[CorpusItem] = []

    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if any(part in IGNORE_DIRS for part in path.relative_to(root).parts[:-1]):
                continue
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                stem = path.stem
                items.append(
                    CorpusItem(
                        source_path=_rel(path, base),
                        abs_path=path,
                        doc_type="paper",
                        tier="core" if stem in settings.core_sources else "breadth",
                        title=_pdf_title(path, stem),
                        phase=None,
                        content_hash=sha256_file(path),
                        is_pdf=True,
                    )
                )
            elif suffix in {".md", ".markdown", ".txt"}:
                doc_type, phase, title = _text_meta(path)
                items.append(
                    CorpusItem(
                        source_path=_rel(path, base),
                        abs_path=path,
                        doc_type=doc_type,
                        tier="breadth",
                        title=title,
                        phase=phase,
                        content_hash=sha256_file(path),
                        is_pdf=False,
                    )
                )
    return items
