"""Download-hardening in acquire.py: believable fingerprints, session reuse, jitter — verify unchanged.

No real network: the httpx client is faked or acquire.download is stubbed.
"""

from __future__ import annotations

from research_kb import acquire

_UA_TOKEN = {"Windows": "Windows NT", "macOS": "Mac OS X", "Linux": "Linux"}


def test_fingerprints_are_internally_consistent():
    for fp in acquire._FINGERPRINTS:
        ua = fp["User-Agent"]
        assert "httpx" not in ua.lower()  # never a library-default UA
        is_chrome = "Chrome/" in ua and "Firefox" not in ua
        # Chrome carries Sec-CH-UA client hints matching its UA; Firefox omits them entirely.
        assert ("sec-ch-ua-platform" in fp) == is_chrome
        for header in ("Accept", "Accept-Language", "Accept-Encoding", "Sec-Fetch-Mode", "Upgrade-Insecure-Requests"):
            assert header in fp
        if is_chrome:
            platform = fp["sec-ch-ua-platform"].strip('"')
            assert _UA_TOKEN[platform] in ua


def test_browser_headers_rotate_over_whole_pool():
    seen = {acquire._browser_headers()["User-Agent"] for _ in range(300)}
    assert seen == {fp["User-Agent"] for fp in acquire._FINGERPRINTS}


def test_referer_derivation():
    assert acquire._referer_for("https://arxiv.org/pdf/2301.00001v2") == "https://arxiv.org/abs/2301.00001v2"
    assert acquire._referer_for("https://eprint.iacr.org/2023/123.pdf") == "https://eprint.iacr.org/"


def test_request_headers_referer_and_sec_fetch_site():
    h = acquire._request_headers(acquire._FINGERPRINTS[0], "https://arxiv.org/pdf/2301.00001")
    assert h["Referer"] == "https://arxiv.org/abs/2301.00001"
    assert h["Sec-Fetch-Site"] == "same-origin"  # referer host == target host


def test_sleep_backoff_is_jittered_and_bounded(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(acquire.time, "sleep", lambda s: slept.append(s))
    for _ in range(30):
        acquire._sleep_backoff(3.0, 1)  # base * 2**1 * uniform(0.5, 1.5) -> [3.0, 9.0)
    assert all(3.0 <= s < 9.0 for s in slept)
    assert len(set(slept)) > 1  # jitter → not a fixed cadence


# --- HTTP paths, with a fake client / stubbed download --------------------------------------------


class _FakeResponse:
    def __init__(self, status: int, content: bytes):
        self.status_code = status
        self.content = content


class _FakeClient:
    def __init__(self, content: bytes = b"%PDF-1.7 fake"):
        self.calls: list[tuple[str, dict]] = []
        self._content = content

    def get(self, url, headers=None):
        self.calls.append((url, headers or {}))
        return _FakeResponse(200, self._content)


class _SequencedClient:
    """Returns a queued sequence of (status, content) responses, one per `.get` call."""

    def __init__(self, responses: list[tuple[int, bytes]]):
        self._responses = list(responses)
        self.calls = 0

    def get(self, url, headers=None):
        self.calls += 1
        status, content = self._responses[min(self.calls - 1, len(self._responses) - 1)]
        return _FakeResponse(status, content)


def test_download_sends_browser_fingerprint(tmp_path, monkeypatch):
    fp = acquire._FINGERPRINTS[0]
    monkeypatch.setattr(acquire, "_browser_headers", lambda: dict(fp))
    client = _FakeClient()
    ok, _ = acquire.download("https://arxiv.org/pdf/2301.00001", tmp_path / "p.pdf", client=client)
    assert ok
    _, headers = client.calls[0]
    assert headers["User-Agent"] == fp["User-Agent"]
    assert "python-httpx" not in headers["User-Agent"]
    assert headers["Referer"] == "https://arxiv.org/abs/2301.00001"
    assert headers["Sec-Fetch-Site"] == "same-origin"


def test_manifest_reuses_one_session(tmp_path, monkeypatch):
    clients: list[int] = []

    def fake_download(url, dest, client=None, **kw):
        clients.append(id(client))
        dest.write_bytes(b"%PDF-1.7 ok")
        return True, "ok"

    monkeypatch.setattr(acquire, "download", fake_download)
    monkeypatch.setattr(acquire, "verify_pdf", lambda p, t, **k: (True, "match"))
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        "- {filename: a.pdf, title: A, url: 'http://x/a'}\n- {filename: b.pdf, title: B, url: 'http://x/b'}\n",
        encoding="utf-8",
    )
    acquire.acquire_from_manifest(manifest, tmp_path / "dest")
    assert len(clients) == 2
    assert clients[0] is not None and clients[0] == clients[1]  # same client object across entries


def test_verification_stays_unconditional(tmp_path, monkeypatch):
    # A friendlier fingerprint changes only *whether we get bytes*, never *whether we trust them*:
    # a downloaded-but-wrong document is still rejected by verify_pdf.
    def fake_download(url, dest, client=None, **kw):
        dest.write_bytes(b"%PDF-1.7 unrelated garbage, not the expected document")
        return True, "ok"

    monkeypatch.setattr(acquire, "download", fake_download)
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        "- {filename: a.pdf, title: 'A Very Specific Unique Title Xyzzy Qwerty', url: 'http://x/a'}\n",
        encoding="utf-8",
    )
    res = acquire.acquire_from_manifest(manifest, tmp_path / "dest")
    assert res.rejected and not res.acquired


# --- Backoff triggers on 429/5xx but never on a deterministic status ------------------------------


def test_fetch_pdf_retries_on_transient_then_succeeds(monkeypatch):
    # 429 (throttled) then 503 (transient) are retried with backoff; the 3rd attempt's PDF wins.
    backoffs: list[int] = []
    monkeypatch.setattr(acquire, "_sleep_backoff", lambda base, attempt: backoffs.append(attempt))
    client = _SequencedClient([(429, b"slow down"), (503, b"unavailable"), (200, b"%PDF-1.7 ok")])
    content, reason = acquire._fetch_pdf(client, "http://x/a", attempts=4, backoff=3.0, headers=acquire._FINGERPRINTS[0])
    assert content == b"%PDF-1.7 ok" and reason == ""
    assert client.calls == 3  # stopped as soon as the PDF arrived
    assert backoffs == [0, 1]  # a backoff between each of the two retries, exponent growing


def test_fetch_pdf_exhausts_attempts_on_persistent_5xx(monkeypatch):
    backoffs: list[int] = []
    monkeypatch.setattr(acquire, "_sleep_backoff", lambda base, attempt: backoffs.append(attempt))
    client = _SequencedClient([(503, b"unavailable")])
    content, reason = acquire._fetch_pdf(client, "http://x/a", attempts=3, backoff=3.0, headers=acquire._FINGERPRINTS[0])
    assert content is None and reason == "HTTP 503"
    assert client.calls == 3  # all attempts spent
    assert backoffs == [0, 1]  # backoff between attempts, none after the last


def test_fetch_pdf_does_not_retry_deterministic_403(monkeypatch):
    # A 403 is a hard block on this route — retrying the same source is pointless, so we break out
    # immediately and let the Wayback fallback take over instead of burning attempts + backoff.
    backoffs: list[int] = []
    monkeypatch.setattr(acquire, "_sleep_backoff", lambda base, attempt: backoffs.append(attempt))
    client = _SequencedClient([(403, b"forbidden")])
    content, reason = acquire._fetch_pdf(client, "http://x/a", attempts=4, backoff=3.0, headers=acquire._FINGERPRINTS[0])
    assert content is None and reason == "HTTP 403"
    assert client.calls == 1  # no retry on a deterministic status
    assert backoffs == []  # and therefore no backoff sleep


# --- Verification stays unconditional: a non-PDF response is rejected by magic bytes --------------


def test_verify_pdf_rejects_non_pdf_magic_bytes(tmp_path):
    junk = tmp_path / "challenge.pdf"
    junk.write_bytes(b"<!DOCTYPE html><html>Access denied</html>")
    ok, reason = acquire.verify_pdf(junk, "A Very Specific Unique Title")
    assert not ok and "not a PDF" in reason
