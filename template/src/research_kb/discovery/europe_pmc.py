"""Europe PMC discovery — the REST search API (``ebi.ac.uk/europepmc/webservices/rest/search``).

The biomedical workhorse, preferred over raw PubMed E-utilities because Europe PMC exposes an
open-access full-text **PDF** link per article (``fullTextUrlList``), so a hit is directly
acquirable. We ask for ``resultType=core`` (the ``lite`` default omits the full-text links) and, for
each result, take a ``documentStyle == "pdf"`` rendition whose availability is open access. Articles
with no open-access PDF are skipped — mirroring the Semantic Scholar provider — because there is no
URL ``acquire`` could fetch and verify; a bare abstract landing page is never handed on. The date
filter threads ``since`` in as a ``PUB_YEAR`` lower bound, Europe PMC's Lucene-style range syntax.
"""

from __future__ import annotations

from datetime import date

import httpx

from .base import Candidate, Provider, register
from .http import get

_API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
# Europe PMC accepts a Lucene range for the publication year; an open-ended "since last run" lower
# bound is written ``PUB_YEAR:[<since> TO <far future>]`` (mirrors arXiv's far-future range stamp).
_FAR_YEAR = "3000"


def _pdf_url(result: dict) -> str | None:
    """An open-access full-text PDF link from the result's ``fullTextUrlList``, else ``None``.

    Europe PMC lists every rendition (pdf/html) with an availability code; we take a
    ``documentStyle == "pdf"`` entry whose ``availabilityCode`` is ``OA`` (open access), so the url
    handed to ``acquire`` is a directly downloadable PDF rather than a paywalled or landing page.
    """
    entries = ((result.get("fullTextUrlList") or {}).get("fullTextUrl")) or []
    for entry in entries:
        if entry.get("documentStyle") == "pdf" and entry.get("availabilityCode") == "OA" and entry.get("url"):
            return str(entry["url"])
    return None


def _identifier(result: dict) -> str:
    """A stable id: the DOI, else the PMC id, else Europe PMC's native record id."""
    return str(result.get("doi") or result.get("pmcid") or result.get("id") or "")


def _year(result: dict) -> int | None:
    year = result.get("pubYear")
    if isinstance(year, int):
        return year
    if isinstance(year, str) and year.isdigit():
        return int(year)
    return None


def _parse(payload: dict) -> list[Candidate]:
    out: list[Candidate] = []
    for result in ((payload.get("resultList") or {}).get("result")) or []:
        pdf = _pdf_url(result)
        title = " ".join((result.get("title") or "").split())
        ident = _identifier(result)
        if not (pdf and title and ident):
            continue  # no open-access PDF (or no title/id) → nothing acquire could fetch
        out.append(
            Candidate(
                provider="europe_pmc",
                id=ident,
                title=title,
                url=pdf,
                year=_year(result),
                doi=result.get("doi"),
            )
        )
    return out


def _build_query(query: str, since: date | None) -> str:
    """The REST ``query``: the seed terms, optionally ANDed with a ``PUB_YEAR`` lower bound."""
    if since is None:
        return query
    return f"({query}) AND (PUB_YEAR:[{since.year} TO {_FAR_YEAR}])"


def search(
    query: str,
    *,
    since: date | None = None,
    limit: int = 50,
    client: httpx.Client | None = None,
) -> list[Candidate]:
    """Search Europe PMC for ``query``; ``since`` bounds by publication year. Up to ``limit`` hits."""
    own = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=30.0)
    params: dict[str, str | int] = {
        "query": _build_query(query, since),
        "format": "json",
        "resultType": "core",  # 'lite' (the default) omits fullTextUrlList — we need the PDF links
        "pageSize": limit,
    }
    try:
        resp = get(client, _API, params=params, accept="application/json")
        return _parse(resp.json())
    finally:
        if own:
            client.close()


register(Provider("europe_pmc", search))
