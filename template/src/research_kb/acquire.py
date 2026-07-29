"""Verified acquisition: download a candidate PDF and *prove* it is the right document.

A naive downloader is easy to fool — a hallucinated URL yields an unrelated PDF or an HTML error
page that is then silently indexed. This module closes that loop: every download is checked to be a
real PDF whose first pages actually match the expected title, so a wrong URL is rejected instead of
indexed. Some hosts block datacenter IPs; a public-archive mirror is used as a fallback route.
Sources that cannot be verified are reported, not kept.
"""

from __future__ import annotations

import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # pymupdf
import httpx
import yaml

_STOP = frozenset({"a", "an", "the", "of", "for", "and", "or", "to", "in", "on", "with", "without", "via", "based"})
# Some PDF hosts 403 datacenter IPs. The Wayback Machine (archive.org) mirrors them and is not behind
# that block; `2id_` returns the latest snapshot's raw bytes (no archive toolbar).
_WAYBACK = "https://web.archive.org/web/2id_/"

# --- Believable browser fingerprints ------------------------------------------------------------
# A single stale User-Agent (or a library-default `python-httpx/…`) is a trivial bot tell, so we
# rotate over a pool of *internally consistent* fingerprints: each bundle's client hints (or their
# deliberate absence, for Firefox) match its User-Agent, so the request looks like the browser it
# claims to be. One fingerprint is chosen per document and kept stable across that document's routes;
# the httpx session (cookies, connections) is reused across documents. Bump these periodically — an
# outdated UA is as much a tell as a missing one.
_CHROME_ACCEPT = (
    "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,"
    "*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
)
_FIREFOX_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"


def _chrome(version: str, ua_platform: str, ch_platform: str) -> dict[str, str]:
    """A Chrome fingerprint whose Sec-CH-UA client hints match its User-Agent."""
    return {
        "User-Agent": (
            f"Mozilla/5.0 ({ua_platform}) AppleWebKit/537.36 (KHTML, like Gecko) "
            f"Chrome/{version}.0.0.0 Safari/537.36"
        ),
        "Accept": _CHROME_ACCEPT,
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "sec-ch-ua": f'"Google Chrome";v="{version}", "Chromium";v="{version}", "Not_A Brand";v="24"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": f'"{ch_platform}"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
    }


def _firefox(version: str, ua_platform: str) -> dict[str, str]:
    """A Firefox fingerprint — Firefox sends no Sec-CH-UA client hints, so we omit them (consistent)."""
    return {
        "User-Agent": f"Mozilla/5.0 ({ua_platform}; rv:{version}.0) Gecko/20100101 Firefox/{version}.0",
        "Accept": _FIREFOX_ACCEPT,
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
    }


_FINGERPRINTS: list[dict[str, str]] = [
    _chrome("133", "Windows NT 10.0; Win64; x64", "Windows"),
    _chrome("132", "Macintosh; Intel Mac OS X 10_15_7", "macOS"),
    _chrome("133", "X11; Linux x86_64", "Linux"),
    _firefox("134", "Windows NT 10.0; Win64; x64"),
    _firefox("133", "Macintosh; Intel Mac OS X 10.15"),
]


def _browser_headers() -> dict[str, str]:
    """A fresh copy of a randomly chosen, internally-consistent browser fingerprint."""
    return dict(random.choice(_FINGERPRINTS))


def _referer_for(url: str) -> str:
    """A plausible page a browser would have been on just before fetching this PDF."""
    m = re.match(r"(https?://arxiv\.org)/pdf/(.+?)(?:\.pdf)?$", url)
    if m:  # arXiv: the abstract page for this PDF
        return f"{m.group(1)}/abs/{m.group(2)}"
    return str(httpx.URL(url).copy_with(path="/", query=None, fragment=None))


def _sleep_backoff(base: float, attempt: int) -> None:
    """Exponential backoff with jitter (no two retries land on the same cadence)."""
    time.sleep(base * (2**attempt) * random.uniform(0.5, 1.5))


@dataclass
class AcquireResult:
    acquired: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)          # already present
    rejected: list[tuple[str, str]] = field(default_factory=list)  # (filename, reason)


def _sig_words(title: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", title.lower()) if len(w) > 2 and w not in _STOP}


def _first_pages_text(path: Path, pages: int = 2) -> str:
    with fitz.open(path) as doc:
        return " ".join(doc[i].get_text("text") for i in range(min(pages, doc.page_count)))


def verify_pdf(path: Path, expected_title: str, min_overlap: float = 0.6) -> tuple[bool, str]:
    """True if ``path`` is a PDF whose first pages contain most significant words of the title."""
    if path.read_bytes()[:4] != b"%PDF":
        return False, "not a PDF (likely an error/HTML page)"
    try:
        text = _first_pages_text(path).lower()
    except Exception as exc:  # noqa: BLE001
        return False, f"unreadable PDF: {exc}"
    words = _sig_words(expected_title)
    if not words:
        return False, "expected title has no significant words"
    present = sum(1 for w in words if w in text)
    ratio = present / len(words)
    if ratio >= min_overlap:
        return True, f"title match {ratio:.0%} ({present}/{len(words)} words)"
    return False, f"content mismatch: only {ratio:.0%} of title words found ({present}/{len(words)})"


def _request_headers(base: dict[str, str], url: str) -> dict[str, str]:
    """The fingerprint plus a Referer and a Sec-Fetch-Site consistent with it."""
    referer = _referer_for(url)
    same_origin = httpx.URL(referer).host == httpx.URL(url).host
    return {**base, "Referer": referer, "Sec-Fetch-Site": "same-origin" if same_origin else "cross-site"}


def _fetch_pdf(
    client: httpx.Client, url: str, attempts: int, backoff: float, headers: dict[str, str]
) -> tuple[bytes | None, str]:
    """GET ``url`` with a believable browser fingerprint, requiring HTTP 200 + PDF magic bytes. Retries
    transient failures (a refused/throttled connection, a 429/5xx) with jittered exponential backoff; a
    deterministic status (e.g. 403) is not retried."""
    req_headers = _request_headers(headers, url)
    reason = ""
    for i in range(attempts):
        try:
            resp = client.get(url, headers=req_headers)
        except Exception as exc:  # noqa: BLE001  (connection refused when the mirror throttles us)
            reason = f"{type(exc).__name__}: {exc}"
        else:
            if resp.status_code == 200 and resp.content[:4] == b"%PDF":
                return resp.content, ""
            reason = f"HTTP {resp.status_code}" if resp.status_code != 200 else "HTTP 200 but not a PDF (challenge page)"
            if resp.status_code not in (429, 500, 502, 503, 504):
                break  # 403 etc. is deterministic — retrying the same source won't help
        if i < attempts - 1:
            _sleep_backoff(backoff, i)
    return None, reason


def download(url: str, dest: Path, timeout: float = 90.0, client: httpx.Client | None = None) -> tuple[bool, str]:
    """Fetch ``url`` to ``dest`` as a PDF, falling back to the Wayback mirror when the origin blocks us.

    Each source must return HTTP 200 *and* PDF magic bytes — a 403 or a 200 HTML challenge page is not
    accepted, it just triggers the next source. The winning route is reported so a Cloudflare bypass is
    visible in the run summary. The Wayback route is paced + retried because archive.org throttles bursts
    (a refused connection mid-run is a rate limit, not a hard block). Pass a shared ``client`` to reuse
    the session (cookies + connections) across documents; one browser fingerprint is chosen per call and
    kept stable across this document's routes.
    """
    own_client = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=timeout)
    headers = _browser_headers()
    # (label, url, attempts, pace_before): pace the mirror to stay under archive.org's burst throttle.
    routes = [("origin", url, 1, 0.0), ("wayback", _WAYBACK + url, 4, 2.0)]
    errors = []
    try:
        for label, candidate, attempts, pace in routes:
            if pace:
                time.sleep(pace)
            content, reason = _fetch_pdf(client, candidate, attempts=attempts, backoff=3.0, headers=headers)
            if content is not None:
                dest.write_bytes(content)
                return True, f"{len(content)} bytes via {label}"
            errors.append(f"{label}: {reason}")
        return False, "; ".join(errors)
    finally:
        if own_client:
            client.close()


def acquire_from_manifest(
    manifest_path: Path,
    dest_dir: Path,
    on_result: Callable[[str, str, str], None] | None = None,
) -> AcquireResult:
    """Download + verify each entry {filename, title, url}. Only verified PDFs land in ``dest_dir``.

    Entries are processed sequentially (deliberately — the Wayback fallback throttles under bursts).
    ``on_result(status, filename, detail)`` is invoked as each entry resolves (status ∈ OK/SKIP/REJ) so a
    caller can stream live progress instead of waiting for the whole run.
    """
    entries = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or []
    dest_dir.mkdir(parents=True, exist_ok=True)
    result = AcquireResult()

    def emit(status: str, filename: str, detail: str) -> None:
        if on_result is not None:
            on_result(status, filename, detail)

    # One session for the whole run: cookies a host sets on first contact persist to later entries.
    with httpx.Client(follow_redirects=True, timeout=90.0) as client:
        for entry in entries:
            filename, title, url = entry["filename"], entry["title"], entry["url"]
            target = dest_dir / filename
            if target.exists():
                result.skipped.append(filename)
                emit("SKIP", filename, "already present")
                continue

            tmp = dest_dir / (filename + ".part")
            ok, info = download(url, tmp, client=client)
            if not ok:
                tmp.unlink(missing_ok=True)
                reason = f"download failed: {info}"
                result.rejected.append((filename, reason))
                emit("REJ", filename, reason)
                continue

            verified, reason = verify_pdf(tmp, title)
            if verified:
                tmp.rename(target)
                result.acquired.append(f"{filename} ({reason})")
                emit("OK", filename, reason)
            else:
                tmp.unlink(missing_ok=True)
                result.rejected.append((filename, reason))
                emit("REJ", filename, reason)
    return result
