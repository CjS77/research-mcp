"""arXiv discovery — the Atom export API (``export.arxiv.org/api/query``).

Query with the seed terms, newest first; optionally bound by submission date so the
incremental-refresh cron fetches only the delta. Returns candidates whose ``url`` is the canonical
``arxiv.org/pdf/<id>`` link the API itself advertises — never a hand-built one. ``acquire`` already
knows how to fetch and verify arXiv PDFs (its ``_referer_for`` derives the abstract page).
"""

from __future__ import annotations

from datetime import date
from xml.etree import ElementTree as ET

import httpx

from .base import Candidate, Provider, register, since_stamp
from .http import get

_API = "http://export.arxiv.org/api/query"
_ATOM = "{http://www.w3.org/2005/Atom}"
# arXiv accepts submittedDate in a range of ``YYYYMMDDHHMM`` stamps; an open-ended "since last run"
# is expressed as ``[<since> TO <far future>]``.
_FAR_FUTURE = "299912312359"


def _build_search_query(query: str, since: date | None) -> str:
    """The provider's ``search_query``: the seed terms, optionally ANDed with a submittedDate range."""
    base = query if ":" in query else f"all:{query}"
    stamp = since_stamp(since, "%Y%m%d")
    if stamp is None:
        return base
    return f"({base}) AND submittedDate:[{stamp}0000 TO {_FAR_FUTURE}]"


def _pdf_url(entry: ET.Element) -> str | None:
    """The PDF link the entry advertises (``<link title="pdf">``), else derived from its id."""
    for link in entry.findall(f"{_ATOM}link"):
        if link.get("title") == "pdf" and link.get("href"):
            return link.get("href")
    ident = entry.findtext(f"{_ATOM}id")  # e.g. http://arxiv.org/abs/2301.00001v1
    if ident:
        return "https://arxiv.org/pdf/" + ident.rsplit("/abs/", 1)[-1]
    return None


def _parse(xml: str) -> list[Candidate]:
    root = ET.fromstring(xml)
    out: list[Candidate] = []
    for entry in root.findall(f"{_ATOM}entry"):
        ident = entry.findtext(f"{_ATOM}id") or ""
        arxiv_id = ident.rsplit("/abs/", 1)[-1]
        title = " ".join((entry.findtext(f"{_ATOM}title") or "").split())
        url = _pdf_url(entry)
        if not (arxiv_id and title and url):
            continue
        published = entry.findtext(f"{_ATOM}published") or ""
        year = int(published[:4]) if published[:4].isdigit() else None
        out.append(Candidate(provider="arxiv", id=arxiv_id, title=title, url=url, year=year))
    return out


def search(
    query: str,
    *,
    since: date | None = None,
    limit: int = 50,
    client: httpx.Client | None = None,
) -> list[Candidate]:
    """Search arXiv for ``query`` (newest first), optionally since ``since``; up to ``limit`` hits."""
    own = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=30.0)
    try:
        resp = get(
            client,
            _API,
            params={
                "search_query": _build_search_query(query, since),
                "start": 0,
                "max_results": limit,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            },
            accept="application/atom+xml",
        )
        return _parse(resp.text)
    finally:
        if own:
            client.close()


register(Provider("arxiv", search))
