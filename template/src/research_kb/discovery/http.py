"""Shared HTTP layer for provider API calls — a believable-browser GET, reusing ``acquire``.

Discovery hits provider *metadata* APIs (Atom/JSON), not PDF hosts, so this is a thin sibling of
:mod:`research_kb.acquire`'s download path rather than a copy of it. It **reuses** acquire's rotating
User-Agent pool (:func:`acquire._browser_headers`) and its jittered exponential backoff
(:func:`acquire._sleep_backoff`) so we never re-roll a fingerprint or ship a ``python-httpx/…``
default UA — the same bot tell the download hardening set out to avoid. The one difference: an API
GET advertises ``Accept: application/json`` (or Atom XML), and drops the ``Sec-Fetch: navigate`` /
``Upgrade-Insecure-Requests`` document-navigation headers that only make sense for a top-level page
load, keeping the request internally consistent with an XHR a browser would actually issue.

Verification is deliberately *not* here: discovery only finds candidates. The bytes of any PDF a
candidate points at are downloaded and verified by :mod:`research_kb.acquire` — this module never
fetches or trusts a document.
"""

from __future__ import annotations

import httpx

from .. import acquire

# Statuses worth a retry: throttling (429) and transient upstream faults (5xx). Anything else
# (404, 400, …) is deterministic — retrying the identical request cannot change the answer.
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class DiscoveryError(RuntimeError):
    """A provider API call failed after exhausting retries (or returned an unusable payload)."""


def _api_headers(accept: str) -> dict[str, str]:
    """A browser fingerprint adapted for an API GET: real UA + client hints, API-appropriate Accept.

    Starts from one of acquire's internally-consistent fingerprints (so the User-Agent and any Chrome
    ``sec-ch-ua`` client hints match), then swaps the page-navigation ``Accept`` for the API media
    type and omits the document-only ``Sec-Fetch``/``Upgrade-Insecure-Requests`` headers.
    """
    fp = acquire._browser_headers()
    headers = {
        "User-Agent": fp["User-Agent"],
        "Accept": accept,
        "Accept-Language": fp["Accept-Language"],
        "Accept-Encoding": fp["Accept-Encoding"],
    }
    # Carry the Chrome client hints when this fingerprint is Chrome; Firefox fingerprints omit them.
    headers.update({h: fp[h] for h in ("sec-ch-ua", "sec-ch-ua-mobile", "sec-ch-ua-platform") if h in fp})
    return headers


def get(
    client: httpx.Client,
    url: str,
    *,
    params: dict[str, str | int] | None = None,
    accept: str = "application/json",
    attempts: int = 3,
    backoff: float = 2.0,
) -> httpx.Response:
    """GET ``url`` with a believable fingerprint, retrying throttles/5xx with jittered backoff.

    Returns the 200 response for the caller to parse (``.json()`` / ``.text``). A deterministic
    non-200 is not retried; on exhaustion (or a persistent transient error) a :class:`DiscoveryError`
    is raised so a provider never silently returns an empty candidate list on a network fault.
    """
    headers = _api_headers(accept)
    reason = ""
    for attempt in range(attempts):
        try:
            resp = client.get(url, params=params, headers=headers)
        except Exception as exc:  # noqa: BLE001  (a refused/timed-out connection is retryable)
            reason = f"{type(exc).__name__}: {exc}"
        else:
            if resp.status_code == 200:
                return resp
            reason = f"HTTP {resp.status_code}"
            if resp.status_code not in _RETRY_STATUS:
                break  # deterministic — retrying the identical request won't help
        if attempt < attempts - 1:
            acquire._sleep_backoff(backoff, attempt)
    raise DiscoveryError(f"{url}: {reason}")
