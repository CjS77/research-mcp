"""Semantic Scholar discovery — the Graph API paper search (``api.semanticscholar.org/graph/v1``).

S2 is the workhorse for the citation-graph expansion wave (step 8 of the playbook) as well as
first-seed discovery: it exposes an ``openAccessPdf`` link per paper, so a hit is directly
acquirable. Papers without an open-access PDF are skipped here — there is no URL to hand ``acquire``
— and their ``externalIds`` (DOI/ArXiv) are the bridge a future citation-resolution step will use to
re-query the other providers. The date filter maps to S2's ``year=<since>-`` (open-ended lower bound).
"""

from __future__ import annotations

from datetime import date

import httpx

from .base import Candidate, Provider, register
from .http import get

_API = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = "title,year,openAccessPdf,externalIds"


def _external_id(paper: dict) -> str:
    """A stable id: the S2 paperId, or a provider-native external id when present."""
    ext = paper.get("externalIds") or {}
    return str(ext.get("DOI") or ext.get("ArXiv") or paper.get("paperId") or "")


def _parse(payload: dict) -> list[Candidate]:
    out: list[Candidate] = []
    for paper in payload.get("data") or []:
        pdf = (paper.get("openAccessPdf") or {}).get("url")
        title = " ".join((paper.get("title") or "").split())
        ident = _external_id(paper)
        if not (pdf and title and ident):
            continue  # no open-access PDF → nothing acquire could fetch
        ext = paper.get("externalIds") or {}
        year = paper.get("year") if isinstance(paper.get("year"), int) else None
        out.append(
            Candidate(
                provider="semantic_scholar",
                id=ident,
                title=title,
                url=str(pdf),
                year=year,
                doi=ext.get("DOI"),
            )
        )
    return out


def search(
    query: str,
    *,
    since: date | None = None,
    limit: int = 50,
    client: httpx.Client | None = None,
) -> list[Candidate]:
    """Search Semantic Scholar for ``query``; ``since`` bounds by publication year. Up to ``limit`` hits."""
    own = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=30.0)
    params: dict[str, str | int] = {"query": query, "limit": limit, "fields": _FIELDS}
    if since is not None:
        params["year"] = f"{since.year}-"
    try:
        resp = get(client, _API, params=params, accept="application/json")
        return _parse(resp.json())
    finally:
        if own:
            client.close()


register(Provider("semantic_scholar", search))
