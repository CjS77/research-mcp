"""Web-source ingestion: pull a remote page into ``reference/`` as first-class Markdown.

Local ``.md``/``.txt`` are already corpus material; the one missing step was fetching a remote
page/standard and converting it to Markdown so it can be dropped in ``reference/`` and indexed like
any local file. This module is that step. Two properties carry over from :mod:`.acquire`:

* **Believable browser fingerprint.** Page fetches reuse acquire's UA pool, per-request header set
  (Referer + ``Sec-Fetch-Site``), session, and jittered backoff — a page fetch looks like the same
  browser a PDF fetch does, not a library-default client.
* **Verification is non-negotiable.** An HTML response has no ``%PDF`` magic to check, so a wrong or
  hostile page (a challenge/CAPTCHA wall, a 404 stub, a JS-only shell) yields little or no main
  content. We extract the *main* content (nav/boilerplate stripped) and reject anything without a
  real title and a non-trivial body — junk is rejected, never indexed.

Downstream is unchanged: the output is an ordinary ``.md`` that ``index --scan reference`` ingests
with no special-casing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx
import trafilatura

from .acquire import _browser_headers, _request_headers, _sleep_backoff

# A 200 page can still be junk (a challenge wall or an error stub returns HTTP 200 with almost no
# real content). Two guards catch it: a minimum body of extracted main-content text, and an explicit
# set of anti-bot / error phrases that mark a page as a wall even when it is otherwise short.
_MIN_CONTENT_CHARS = 200
_CHALLENGE_MARKERS: tuple[str, ...] = (
    "access denied",
    "attention required",
    "are you a robot",
    "captcha",
    "enable javascript",
    "just a moment",
    "please verify you are a human",
    "verify you are human",
)
_TITLE_TAG_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


@dataclass
class FetchOutcome:
    """The result of one page fetch. ``status`` mirrors acquire's OK/SKIP/REJ vocabulary."""

    status: str  # "OK" | "SKIP" | "REJ"
    detail: str
    path: Path | None = None


def _fetch_html(
    client: httpx.Client, url: str, attempts: int, backoff: float, headers: dict[str, str]
) -> tuple[str | None, str]:
    """GET ``url`` with a believable browser fingerprint, requiring HTTP 200; return the decoded body.

    Retries a transient failure (refused connection, 429/5xx) with jittered exponential backoff;
    a deterministic status (e.g. 403/404) is not retried. Content is *not* trusted here — that is the
    verifier's job — only the transport is handled.
    """
    req_headers = _request_headers(headers, url)
    reason = ""
    for i in range(attempts):
        try:
            resp = client.get(url, headers=req_headers)
        except Exception as exc:  # noqa: BLE001  (a refused/throttled connection is retryable)
            reason = f"{type(exc).__name__}: {exc}"
        else:
            if resp.status_code == 200:
                return resp.text, ""
            reason = f"HTTP {resp.status_code}"
            if resp.status_code not in (429, 500, 502, 503, 504):
                break  # deterministic — retrying the same URL won't help
        if i < attempts - 1:
            _sleep_backoff(backoff, i)
    return None, reason


def _title_from_html(html: str) -> str:
    """Best-effort page title: trafilatura metadata first, the ``<title>`` tag as a fallback."""
    try:
        meta = trafilatura.extract_metadata(html)
    except Exception:  # noqa: BLE001  (metadata parsing is best-effort)
        meta = None
    if meta is not None and getattr(meta, "title", None):
        return str(meta.title).strip()
    m = _TITLE_TAG_RE.search(html)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def extract_markdown(html: str) -> tuple[str, str]:
    """Convert a raw HTML page to (main-content Markdown, title).

    Uses trafilatura to keep the article body and drop nav/boilerplate. Returns empty strings when
    there is no extractable main content (a challenge/JS-only shell), which the verifier then rejects.
    """
    body = trafilatura.extract(html, output_format="markdown", include_comments=False, include_tables=True)
    return (body or "").strip(), _title_from_html(html)


def verify_page(title: str, markdown: str) -> tuple[bool, str]:
    """True if the extracted page is a real document, not a wall/error/stub. Verification is unconditional.

    Mirrors the intent of :func:`acquire.verify_pdf`: with no ``%PDF`` magic to lean on, a page is
    trusted only when it has a title *and* a non-trivial body of main content, and shows none of the
    anti-bot/error markers of a challenge page.
    """
    text = markdown.strip()
    if not text:
        return False, "no extractable main content (challenge page or JS-only shell)"
    haystack = f"{title}\n{text[:500]}".lower()
    if len(text) < 800 and any(marker in haystack for marker in _CHALLENGE_MARKERS):
        return False, "looks like a challenge/error page (anti-bot marker + little content)"
    if len(text) < _MIN_CONTENT_CHARS:
        return False, f"content too short ({len(text)} chars; likely an error/stub page)"
    if not title:
        return False, "no page title found (likely not a real document)"
    return True, f"{len(text)} chars extracted"


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:80]


def _stem_for(url: str, title: str) -> str:
    """A stable, filesystem-safe stem from the title, falling back to the URL path."""
    if title and (slug := _slugify(title)):
        return slug
    path = httpx.URL(url).path.strip("/")
    return _slugify(path.replace("/", "-")) or "page"


def _render(title: str, url: str, markdown: str) -> str:
    """Compose the corpus file: an H1 title, a provenance line, then the extracted body.

    The H1 is what ``corpus._text_meta`` reads back as the document title, and the ``> Source`` line
    keeps the origin URL visible in the indexed text. A duplicate leading H1 from the extractor is
    dropped so the title appears once.
    """
    body = markdown
    if title and body.split("\n", 1)[0].strip() == f"# {title}":
        body = body.split("\n", 1)[1].lstrip("\n") if "\n" in body else ""
    head = f"# {title}\n\n" if title else ""
    return f"{head}> Source: <{url}>  \n> Fetched: {date.today().isoformat()}\n\n{body}\n"


def fetch_page(
    url: str,
    dest_dir: Path,
    *,
    stem: str | None = None,
    force: bool = False,
    timeout: float = 90.0,
    client: httpx.Client | None = None,
) -> FetchOutcome:
    """Fetch ``url``, convert to Markdown, verify, and write it into ``dest_dir`` as a ``.md``.

    Reuses acquire's browser fingerprint for the fetch and rejects a page that fails verification
    (challenge/error/near-empty) instead of writing it. An existing target is skipped unless
    ``force``. Pass a shared ``client`` to reuse the session across several fetches.
    """
    own_client = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=timeout)
    headers = _browser_headers()
    try:
        text, reason = _fetch_html(client, url, attempts=3, backoff=3.0, headers=headers)
        if text is None:
            return FetchOutcome("REJ", f"fetch failed: {reason}")

        markdown, title = extract_markdown(text)
        ok, detail = verify_page(title, markdown)
        if not ok:
            return FetchOutcome("REJ", detail)

        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / f"{stem or _stem_for(url, title)}.md"
        if target.exists() and not force:
            return FetchOutcome("SKIP", "already present (use --force to refetch)", target)

        target.write_text(_render(title, url, markdown), encoding="utf-8")
        return FetchOutcome("OK", detail, target)
    finally:
        if own_client:
            client.close()
