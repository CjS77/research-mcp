"""IACR ePrint discovery — the Cryptology ePrint Archive search (``eprint.iacr.org/search``).

ePrint publishes no JSON search API: its search endpoint returns server-rendered HTML, and the two
other programmatic clients in the wild either scrape that page or filter the ~40-item RSS feed. We
query the search — a real, archive-wide keyword search — and pull ``(id, title)`` pairs out of its
regular result markup with the stdlib :class:`~html.parser.HTMLParser` (no new dependency). Each
result carries the paper's ``<year>/<num>`` id; the canonical PDF is that id's ``.pdf`` sibling
(``https://eprint.iacr.org/<year>/<num>.pdf``), which is exactly what ``acquire`` fetches — and since
ePrint's PDF host Cloudflare-403s datacenter IPs, ``acquire`` already falls back to the Wayback
mirror. Discovery returns the canonical URL; acquire owns the fetch route and verification.

The date filter is year-granular: ePrint's ``submittedafter`` field takes a *year* (``min=1996``,
not a date), so ``since`` maps to ``submittedafter=<since.year>`` — a genuine server-side bound, so
the provider is ``supports_since=True``. The search returns one page (~100 hits); ``limit`` slices it.
"""

from __future__ import annotations

from datetime import date
from html.parser import HTMLParser

import httpx

from .base import Candidate, Provider, register
from .http import get

_API = "https://eprint.iacr.org/search"


class _ResultParser(HTMLParser):
    """Pull ``(id, title)`` pairs out of ePrint's search-result markup.

    Each result opens with ``<a class="paperlink" title="<year>/<num>" href="/<year>/<num>">`` — the
    id anchor — and the first ``<strong>`` after it holds the title. We latch the id on the paperlink
    and take the text of that next ``<strong>`` as the title, collapsing internal whitespace.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[tuple[str, str]] = []
        self._pending_id: str | None = None
        self._in_title = False
        self._title: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        if tag == "a" and "paperlink" in (attr.get("class") or ""):
            self._pending_id = (attr.get("title") or (attr.get("href") or "").strip("/")) or None
        elif tag == "strong" and self._pending_id and not self._in_title:
            self._in_title = True
            self._title = []

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "strong" and self._in_title:
            title = " ".join("".join(self._title).split())
            if self._pending_id and title:
                self.results.append((self._pending_id, title))
            self._pending_id = None
            self._in_title = False


def _pdf_url(paper_id: str) -> str:
    """The canonical ePrint PDF for ``<year>/<num>`` — the id's ``.pdf`` sibling (acquire verifies)."""
    return f"https://eprint.iacr.org/{paper_id}.pdf"


def _year(paper_id: str) -> int | None:
    """The submission year encoded in the id's ``<year>/<num>`` prefix."""
    head = paper_id.split("/", 1)[0]
    return int(head) if head.isdigit() else None


def _parse(html: str, limit: int) -> list[Candidate]:
    parser = _ResultParser()
    parser.feed(html)
    return [
        Candidate(provider="eprint", id=paper_id, title=title, url=_pdf_url(paper_id), year=_year(paper_id))
        for paper_id, title in parser.results[:limit]
    ]


def search(
    query: str,
    *,
    since: date | None = None,
    limit: int = 50,
    client: httpx.Client | None = None,
) -> list[Candidate]:
    """Search IACR ePrint for ``query``; ``since`` bounds by submission *year*. Up to ``limit`` hits."""
    own = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=30.0)
    params: dict[str, str | int] = {"q": query}
    if since is not None:
        params["submittedafter"] = since.year
    try:
        resp = get(client, _API, params=params, accept="text/html")
        return _parse(resp.text, limit)
    finally:
        if own:
            client.close()


register(Provider("eprint", search))
