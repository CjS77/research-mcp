"""Shared source-discovery library: providers → verified-acquisition manifest.

Turns a topic query into candidate ``{title, id, url}`` triples via a registry of provider API
clients (arXiv, Crossref, Semantic Scholar, IACR ePrint today; more register the same way), then emits a manifest
:mod:`research_kb.acquire` downloads and *verifies*. Discovery finds candidates; acquire owns the
non-negotiable verification (200 + ``%PDF`` + ≥60% title overlap). The same code path serves both
first-seed discovery (:func:`discover`) and the incremental-refresh cron (:func:`refresh`), so an
instance never hand-rolls this again.

Importing this package registers the shipped providers (side-effect imports below), so
``provider_names()`` is populated as soon as ``research_kb.discovery`` is imported.
"""

from __future__ import annotations

from datetime import date

import httpx

from ..config import Settings, get_settings

# Side-effect imports: each provider module calls register(...) at import time, so importing the
# package populates the registry. Kept first so provider_names() is ready for anything below.
from . import arxiv, crossref, eprint, europe_pmc, semantic_scholar  # noqa: F401  (registration side effects)
from .base import (
    Candidate,
    Provider,
    get_provider,
    manifest_entries,
    provider_names,
    register,
    write_manifest,
)
from .http import DiscoveryError
from .state import load_last_run, save_last_run

__all__ = [
    "Candidate",
    "Provider",
    "DiscoveryError",
    "discover",
    "refresh",
    "get_provider",
    "provider_names",
    "register",
    "manifest_entries",
    "write_manifest",
    "load_last_run",
    "save_last_run",
]


def discover(
    query: str,
    providers: list[str],
    *,
    since: date | None = None,
    limit: int = 50,
    client: httpx.Client | None = None,
) -> list[Candidate]:
    """Run each named provider's ``search`` for ``query`` and concatenate the candidates (deduped).

    A single httpx session is reused across providers (cookies + connections) unless a ``client`` is
    supplied. An unknown provider name is a hard error rather than a silent empty result. Candidates
    are deduped on ``(provider, id)``, preserving first-seen order.
    """
    own = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=30.0)
    seen: set[tuple[str, str]] = set()
    out: list[Candidate] = []
    try:
        for name in providers:
            provider = get_provider(name)
            if provider is None:
                raise DiscoveryError(f"unknown provider {name!r} (known: {', '.join(provider_names())})")
            for candidate in provider.search(query, since=since, limit=limit, client=client):
                key = (candidate.provider, candidate.id)
                if key not in seen:
                    seen.add(key)
                    out.append(candidate)
        return out
    finally:
        if own:
            client.close()


def refresh(
    query: str,
    providers: list[str],
    settings: Settings | None = None,
    *,
    limit: int = 50,
    client: httpx.Client | None = None,
    today: date | None = None,
) -> list[Candidate]:
    """Incremental discovery: fetch only each provider's delta since its persisted last-run date.

    Reuses :func:`discover` per provider so first-seed and cron share one code path; advances each
    provider's last-run marker to ``today`` after a successful run. ``today`` is injectable for tests.
    """
    settings = settings or get_settings()
    now = today or date.today()
    own = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=30.0)
    out: list[Candidate] = []
    try:
        for name in providers:
            since = load_last_run(settings, name)
            out.extend(discover(query, [name], since=since, limit=limit, client=client))
            save_last_run(settings, name, now)
        return out
    finally:
        if own:
            client.close()
