"""Web-source ingestion (fetch -> Markdown): main-content extraction, verification, fingerprint reuse.

No real network: the httpx client is faked and canned HTML is fed straight through.
"""

from __future__ import annotations

from research_kb import acquire, fetch
from research_kb.config import Settings
from research_kb.corpus import scan

# --- Canned pages ---------------------------------------------------------------------------------

_ARTICLE_HTML = """<html><head><title>Widget Standard v2</title></head>
<body>
  <nav>Home About Contact Login Menu Subscribe</nav>
  <article>
    <h1>Widget Standard v2</h1>
    <p>A widget MUST implement the frobnicate method. This is the core requirement of the standard
       and applies to every conforming implementation without exception whatsoever.</p>
    <p>Section two covers the wibble protocol in exhaustive and painstaking detail across a great
       many carefully worded normative paragraphs, none of which may be skipped.</p>
  </article>
  <footer>Copyright 2026 Boilerplate Incorporated. All rights reserved.</footer>
</body></html>"""

_CHALLENGE_HTML = """<html><head><title>Just a moment...</title></head>
<body><h1>Just a moment...</h1><p>Please verify you are a human. Enable JavaScript and
cookies to continue.</p></body></html>"""

_EMPTY_HTML = "<html><head><title>Nothing</title></head><body><div id='app'></div></body></html>"


class _FakeResponse:
    def __init__(self, status: int, text: str):
        self.status_code = status
        self.text = text


class _FakeClient:
    """Records requests and returns a fixed (status, body); optional per-call sequence for retries."""

    def __init__(self, status: int = 200, text: str = _ARTICLE_HTML, sequence=None):
        self.calls: list[tuple[str, dict]] = []
        self._status = status
        self._text = text
        self._sequence = list(sequence) if sequence else None

    def get(self, url, headers=None):
        self.calls.append((url, headers or {}))
        if self._sequence is not None:
            status, text = self._sequence[min(len(self.calls) - 1, len(self._sequence) - 1)]
            return _FakeResponse(status, text)
        return _FakeResponse(self._status, self._text)


# --- Extraction: main content only ----------------------------------------------------------------


def test_extract_markdown_keeps_main_content_and_drops_boilerplate():
    md, title = fetch.extract_markdown(_ARTICLE_HTML)
    assert title == "Widget Standard v2"
    assert "frobnicate" in md and "wibble protocol" in md
    assert "Login" not in md and "Boilerplate" not in md  # nav + footer stripped


def test_verify_page_accepts_real_document():
    md, title = fetch.extract_markdown(_ARTICLE_HTML)
    ok, _ = fetch.verify_page(title, md)
    assert ok


def test_verify_page_rejects_challenge_wall():
    md, title = fetch.extract_markdown(_CHALLENGE_HTML)
    ok, reason = fetch.verify_page(title, md)
    assert not ok and reason


def test_verify_page_rejects_empty_shell():
    md, title = fetch.extract_markdown(_EMPTY_HTML)
    ok, reason = fetch.verify_page(title, md)
    assert not ok and reason


# --- fetch_page: write a first-class .md into reference/ ------------------------------------------


def test_fetch_page_writes_markdown(tmp_path):
    out = fetch.fetch_page("https://example.org/std/widget", tmp_path / "reference", client=_FakeClient())
    assert out.status == "OK"
    assert out.path is not None and out.path.parent == tmp_path / "reference"
    body = out.path.read_text(encoding="utf-8")
    assert body.startswith("# Widget Standard v2")  # H1 corpus._text_meta reads back as the title
    assert "> Source: <https://example.org/std/widget>" in body  # provenance kept in the indexed text
    assert "frobnicate" in body
    assert body.count("# Widget Standard v2") == 1  # extractor's duplicate leading H1 dropped


def test_fetch_page_filename_derives_from_title(tmp_path):
    out = fetch.fetch_page("https://example.org/x", tmp_path / "ref", client=_FakeClient())
    assert out.path is not None and out.path.name == "widget-standard-v2.md"


def test_fetch_page_honours_explicit_stem(tmp_path):
    out = fetch.fetch_page("https://example.org/x", tmp_path / "ref", stem="rfc-9999", client=_FakeClient())
    assert out.path is not None and out.path.name == "rfc-9999.md"


def test_fetch_page_rejects_challenge_and_writes_nothing(tmp_path):
    dest = tmp_path / "ref"
    out = fetch.fetch_page("https://example.org/x", dest, client=_FakeClient(text=_CHALLENGE_HTML))
    assert out.status == "REJ" and out.path is None
    assert not dest.exists() or not list(dest.glob("*.md"))


def test_fetch_page_rejects_non_200(tmp_path):
    out = fetch.fetch_page("https://example.org/gone", tmp_path / "ref", client=_FakeClient(status=404))
    assert out.status == "REJ" and "404" in out.detail


def test_fetch_page_skips_existing_unless_forced(tmp_path):
    dest = tmp_path / "ref"
    first = fetch.fetch_page("https://example.org/x", dest, client=_FakeClient())
    assert first.status == "OK"
    again = fetch.fetch_page("https://example.org/x", dest, client=_FakeClient())
    assert again.status == "SKIP"
    forced = fetch.fetch_page("https://example.org/x", dest, force=True, client=_FakeClient())
    assert forced.status == "OK"


# --- Fingerprint reuse: page fetches carry acquire's believable browser headers -------------------


def test_fetch_uses_browser_fingerprint(tmp_path, monkeypatch):
    fp = acquire._FINGERPRINTS[0]
    monkeypatch.setattr(fetch, "_browser_headers", lambda: dict(fp))
    client = _FakeClient()
    fetch.fetch_page("https://example.org/std/widget", tmp_path / "ref", client=client)
    _, headers = client.calls[0]
    assert headers["User-Agent"] == fp["User-Agent"]
    assert "python-httpx" not in headers["User-Agent"]
    assert headers["Referer"] and headers["Sec-Fetch-Site"]  # per-request headers applied


def test_fetch_retries_transient_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "_sleep_backoff", lambda base, attempt: None)
    client = _FakeClient(sequence=[(503, "busy"), (200, _ARTICLE_HTML)])
    out = fetch.fetch_page("https://example.org/x", tmp_path / "ref", client=client)
    assert out.status == "OK" and len(client.calls) == 2


def test_fetch_does_not_retry_deterministic_status(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "_sleep_backoff", lambda base, attempt: None)
    client = _FakeClient(sequence=[(403, "forbidden")])
    out = fetch.fetch_page("https://example.org/x", tmp_path / "ref", client=client)
    assert out.status == "REJ" and len(client.calls) == 1  # 403 is not retried


# --- Downstream: the fetched file is a first-class corpus source, no special-casing ---------------


def test_fetched_markdown_is_picked_up_by_corpus_scan(settings: Settings):
    out = fetch.fetch_page("https://example.org/std/widget", settings.reference_dir, client=_FakeClient())
    assert out.status == "OK" and out.path is not None
    item = next(i for i in scan(settings) if i.abs_path == out.path)
    assert item.title == "Widget Standard v2"  # H1 read back as the document title
    assert not item.is_pdf and item.doc_type == "research"
