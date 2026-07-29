"""Crossref discovery — the works API (``api.crossref.org/works``).

Broad DOI-metadata search across publishers. A ``mailto`` (the Crossref "polite pool" courtesy)
routes us onto the faster, more reliable tier when supplied. Each work's ``url`` prefers a
publisher-advertised ``application/pdf`` link; absent one it falls back to the DOI resolver URL — a
landing page, which ``acquire`` will honestly reject as non-PDF rather than index. Discovery returns
what the API gives; it does not guess.
"""

from __future__ import annotations

from datetime import date

import httpx

from .base import Candidate, Provider, register, since_stamp
from .http import get

_API = "https://api.crossref.org/works"


def _pdf_or_landing(item: dict) -> str | None:
    """A publisher ``application/pdf`` link when present, else the DOI resolver URL."""
    for link in item.get("link", []):
        if link.get("content-type") == "application/pdf" and link.get("URL"):
            return str(link["URL"])
    url = item.get("URL") or (f"https://doi.org/{item['DOI']}" if item.get("DOI") else None)
    return str(url) if url else None


def _year(item: dict) -> int | None:
    for key in ("published", "published-print", "published-online", "issued", "created"):
        parts = (item.get(key) or {}).get("date-parts") or []
        if parts and parts[0] and isinstance(parts[0][0], int):
            return parts[0][0]
    return None


def _parse(payload: dict) -> list[Candidate]:
    out: list[Candidate] = []
    for item in (payload.get("message") or {}).get("items", []):
        doi = item.get("DOI")
        titles = item.get("title") or []
        title = " ".join(" ".join(titles).split())
        url = _pdf_or_landing(item)
        if not (doi and title and url):
            continue
        out.append(Candidate(provider="crossref", id=doi, title=title, url=url, year=_year(item), doi=doi))
    return out


def search(
    query: str,
    *,
    since: date | None = None,
    limit: int = 50,
    client: httpx.Client | None = None,
    mailto: str | None = None,
) -> list[Candidate]:
    """Search Crossref for ``query``, optionally since ``since`` (by publication date); up to ``limit``."""
    own = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=30.0)
    params: dict[str, str | int] = {"query": query, "rows": limit}
    stamp = since_stamp(since, "%Y-%m-%d")
    if stamp is not None:
        params["filter"] = f"from-pub-date:{stamp}"
    if mailto:
        params["mailto"] = mailto
    try:
        resp = get(client, _API, params=params, accept="application/json")
        return _parse(resp.json())
    finally:
        if own:
            client.close()


register(Provider("crossref", search))
