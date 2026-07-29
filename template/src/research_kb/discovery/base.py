"""The discovery contract: a ``Candidate``, a ``Provider``, and their registry.

Mirrors :mod:`research_kb.extract.backends`: providers are looked up by name from a module-level
registry, so adding a source (IACR ePrint, PubMed, …) means *registering a sibling module*, not
editing a dispatcher. A provider's only job is to turn a query into candidate ``{title, id, url}``
triples straight from its API — it never guesses a PDF URL and never downloads anything. The URL a
candidate carries is fed verbatim to :mod:`research_kb.acquire`, which downloads and *verifies* the
bytes (200 + ``%PDF`` + title overlap). Discovery finds; acquire verifies; the two never merge.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class Candidate:
    """One discovered document: enough to acquire + cite it, straight from the provider API.

    ``url`` is the acquisition target handed to :mod:`research_kb.acquire` — a direct PDF link when
    the provider gives one, else a landing page (acquire will reject a non-PDF, by design). ``id`` is
    the provider-native identifier (arXiv id, DOI, S2 paperId); ``(provider, id)`` is the dedup key.
    """

    provider: str
    id: str
    title: str
    url: str
    year: int | None = None
    doi: str | None = None
    extra: dict[str, str] = field(default_factory=dict)

    def filename(self) -> str:
        """A stable, filesystem-safe ``<provider>-<id>.pdf`` name for the manifest / reference/ file."""
        stem = _UNSAFE.sub("_", f"{self.provider}-{self.id}").strip("_")
        return f"{stem}.pdf"

    def manifest_entry(self) -> dict[str, str]:
        """The ``{filename, title, url}`` triple ``acquire`` consumes — no more, no less."""
        return {"filename": self.filename(), "title": self.title, "url": self.url}


# A provider's search signature is ``(query, *, since=None, limit=..., client=None) -> list[Candidate]``.
# Typed loosely (``Callable[..., list[Candidate]]``) like the backend registry, since the keyword-only
# shape is a convention the providers share rather than something the type system enforces here.
SearchFn = Callable[..., list[Candidate]]


@dataclass(frozen=True)
class Provider:
    """One named discovery source: its ``search`` callable plus the media type its API speaks."""

    name: str
    search: SearchFn
    supports_since: bool = True


_REGISTRY: dict[str, Provider] = {}


def register(provider: Provider) -> None:
    """Add (or replace) a provider in the registry."""
    _REGISTRY[provider.name] = provider


def get_provider(name: str) -> Provider | None:
    """Look up a provider by name; ``None`` if unknown."""
    return _REGISTRY.get(name)


def provider_names() -> list[str]:
    """Registered provider names, for help text and diagnostics."""
    return sorted(_REGISTRY)


# --- Manifest emission -----------------------------------------------------------------------------


def manifest_entries(candidates: Iterable[Candidate]) -> list[dict[str, str]]:
    """Candidates → the manifest list ``acquire_from_manifest`` reads, deduped by filename."""
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for c in candidates:
        entry = c.manifest_entry()
        if entry["filename"] not in seen:
            seen.add(entry["filename"])
            out.append(entry)
    return out


def write_manifest(candidates: Iterable[Candidate], path: Path, *, merge: bool = True) -> tuple[int, int]:
    """Write (or merge into) an ``acquire``-compatible manifest at ``path``.

    With ``merge`` (the default) existing entries are kept and only genuinely new filenames are
    appended — so an incremental-refresh run grows the manifest with the delta instead of clobbering
    the curated set. Returns ``(added, total)``.
    """
    existing: list[dict[str, str]] = []
    if merge and path.exists():
        existing = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    have = {e["filename"] for e in existing}
    added = [e for e in manifest_entries(candidates) if e["filename"] not in have]
    combined = existing + added
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(combined, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return len(added), len(combined)


def since_stamp(since: date | None, fmt: str) -> str | None:
    """Format a ``since`` date for a provider's API (``None`` → ``None``), e.g. ``%Y%m%d`` / ``%Y-%m-%d``."""
    return since.strftime(fmt) if since is not None else None
