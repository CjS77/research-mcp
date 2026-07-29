"""Discovery library: providers parse canned API payloads into Candidates + manifest entries.

Fully offline — every provider is fed a fake httpx client returning a canned response, so no test
touches the network. Coverage: the registry, each provider's parse + date-filter passthrough, the
shared HTTP retry helper, manifest merge, and incremental-refresh last-run persistence.
"""

from __future__ import annotations

from datetime import date

import pytest
import yaml

from research_kb import discovery
from research_kb.config import Settings
from research_kb.discovery import arxiv, base, crossref, http, semantic_scholar
from research_kb.discovery.base import Candidate, Provider

# --- Fake HTTP -------------------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status: int = 200, text: str = "", json_data: object = None):
        self.status_code = status
        self.text = text
        self._json = json_data

    def json(self) -> object:
        return self._json


class _FakeClient:
    """Serves queued responses (the last repeats) and records every GET's params + headers."""

    def __init__(self, *responses: _FakeResponse):
        self._responses = list(responses)
        self.calls: list[tuple[str, dict, dict]] = []

    def get(self, url, params=None, headers=None):
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        return self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]

    def close(self) -> None:
        pass


# --- Canned payloads -------------------------------------------------------------------------------

_ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.00001v2</id>
    <title>Lattice-Based
      Signatures</title>
    <published>2023-01-05T00:00:00Z</published>
    <link title="pdf" href="http://arxiv.org/pdf/2301.00001v2"/>
    <link title="doi" href="http://dx.doi.org/10.0/x"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2208.09999v1</id>
    <title>Ring-LWE Revisited</title>
    <published>2022-08-20T00:00:00Z</published>
  </entry>
</feed>"""

_CROSSREF_JSON = {
    "message": {
        "items": [
            {
                "DOI": "10.1145/1234.5678",
                "title": ["A Study of Verifiable Computation"],
                "issued": {"date-parts": [[2021, 6]]},
                "URL": "https://doi.org/10.1145/1234.5678",
                "link": [
                    {"URL": "https://example.org/paper.pdf", "content-type": "application/pdf"},
                    {"URL": "https://example.org/paper.xml", "content-type": "application/xml"},
                ],
            },
            {
                "DOI": "10.1000/landing",
                "title": ["Only A Landing Page"],
                "published-print": {"date-parts": [[2019]]},
                "URL": "https://doi.org/10.1000/landing",
            },
            {"DOI": "10.1/no-title", "title": [], "URL": "https://doi.org/10.1/no-title"},
        ]
    }
}

_S2_JSON = {
    "data": [
        {
            "paperId": "abc123",
            "title": "Succinct Arguments",
            "year": 2020,
            "openAccessPdf": {"url": "https://oa.example/succinct.pdf"},
            "externalIds": {"DOI": "10.5/succinct", "ArXiv": "2001.00001"},
        },
        {
            "paperId": "def456",
            "title": "No Open Access Here",
            "year": 2021,
            "openAccessPdf": None,
            "externalIds": {"DOI": "10.5/closed"},
        },
        {
            "paperId": "ghi789",
            "title": "Only A PaperId",
            "year": 2022,
            "openAccessPdf": {"url": "https://oa.example/onlyid.pdf"},
            "externalIds": {},
        },
    ]
}


# --- Registry --------------------------------------------------------------------------------------


def test_shipped_providers_registered():
    assert {"arxiv", "crossref", "semantic_scholar"} <= set(discovery.provider_names())
    assert discovery.get_provider("nonesuch") is None
    assert discovery.get_provider("arxiv") is not None


# --- arXiv -----------------------------------------------------------------------------------------


def test_arxiv_parses_entries_and_manifest():
    client = _FakeClient(_FakeResponse(text=_ARXIV_FEED))
    cands = arxiv.search("lattice", client=client, limit=2)
    assert [c.id for c in cands] == ["2301.00001v2", "2208.09999v1"]
    first = cands[0]
    assert first.title == "Lattice-Based Signatures"  # whitespace/newlines collapsed
    assert first.url == "http://arxiv.org/pdf/2301.00001v2"  # the advertised pdf link
    assert first.year == 2023
    assert first.manifest_entry() == {
        "filename": "arxiv-2301.00001v2.pdf",
        "title": "Lattice-Based Signatures",
        "url": "http://arxiv.org/pdf/2301.00001v2",
    }
    # second entry has no <link title="pdf"> — url is derived from its abstract id
    assert cands[1].url == "https://arxiv.org/pdf/2208.09999v1"
    # request shape: seed terms wrapped as all:, newest-first
    _, params, _ = client.calls[0]
    assert params["search_query"] == "all:lattice"
    assert params["max_results"] == 2
    assert params["sortBy"] == "submittedDate"


def test_arxiv_date_filter_passthrough():
    client = _FakeClient(_FakeResponse(text=_ARXIV_FEED))
    arxiv.search("lattice", since=date(2022, 1, 1), client=client)
    _, params, _ = client.calls[0]
    assert "submittedDate:[202201010000 TO" in params["search_query"]


# --- Crossref --------------------------------------------------------------------------------------


def test_crossref_prefers_pdf_then_landing():
    client = _FakeClient(_FakeResponse(json_data=_CROSSREF_JSON))
    cands = crossref.search("verifiable computation", client=client)
    # the title-less item is dropped; two valid ones remain
    assert [c.id for c in cands] == ["10.1145/1234.5678", "10.1000/landing"]
    assert cands[0].url == "https://example.org/paper.pdf"  # application/pdf link wins
    assert cands[0].year == 2021
    assert cands[0].doi == "10.1145/1234.5678"
    assert cands[1].url == "https://doi.org/10.1000/landing"  # falls back to DOI resolver
    assert cands[1].manifest_entry()["filename"] == "crossref-10.1000_landing.pdf"


def test_crossref_date_and_mailto_passthrough():
    client = _FakeClient(_FakeResponse(json_data=_CROSSREF_JSON))
    crossref.search("x", since=date(2020, 1, 1), client=client, mailto="kb@example.org")
    _, params, _ = client.calls[0]
    assert params["filter"] == "from-pub-date:2020-01-01"
    assert params["mailto"] == "kb@example.org"


# --- Semantic Scholar ------------------------------------------------------------------------------


def test_semantic_scholar_skips_closed_access():
    client = _FakeClient(_FakeResponse(json_data=_S2_JSON))
    cands = semantic_scholar.search("succinct arguments", since=date(2019, 3, 1), client=client)
    # the closed-access paper (no openAccessPdf) is skipped; two acquirable ones remain
    assert [c.id for c in cands] == ["10.5/succinct", "ghi789"]
    assert cands[0].url == "https://oa.example/succinct.pdf"
    assert cands[0].doi == "10.5/succinct"
    assert cands[1].id == "ghi789"  # no external id → falls back to the S2 paperId
    _, params, _ = client.calls[0]
    assert params["year"] == "2019-"


# --- Shared HTTP helper ----------------------------------------------------------------------------


def test_api_headers_look_like_a_browser():
    h = http._api_headers("application/json")
    assert "httpx" not in h["User-Agent"].lower()  # never a library-default UA
    assert h["Accept"] == "application/json"
    # API GET drops the document-navigation headers that only make sense for a page load
    assert "Sec-Fetch-Dest" not in h
    assert "Upgrade-Insecure-Requests" not in h


def test_http_get_retries_transient_then_succeeds(monkeypatch):
    monkeypatch.setattr(http.acquire, "_sleep_backoff", lambda *a: None)
    client = _FakeClient(_FakeResponse(status=503), _FakeResponse(status=200, text="ok"))
    resp = http.get(client, "https://api.example/x")
    assert resp.status_code == 200 and resp.text == "ok"
    assert len(client.calls) == 2  # retried once


def test_http_get_does_not_retry_deterministic(monkeypatch):
    monkeypatch.setattr(http.acquire, "_sleep_backoff", lambda *a: None)
    client = _FakeClient(_FakeResponse(status=404))
    with pytest.raises(http.DiscoveryError):
        http.get(client, "https://api.example/x")
    assert len(client.calls) == 1  # 404 is deterministic — not retried


# --- discover() orchestration ----------------------------------------------------------------------


def _stub_provider(name: str, out: list[Candidate], captured: dict | None = None) -> Provider:
    def search(query, *, since=None, limit=50, client=None):
        if captured is not None:
            captured["since"] = since
            captured["client"] = client
        return out

    return Provider(name, search)


def test_discover_dedups_and_reuses_one_client(monkeypatch):
    dup = Candidate("stub", "1", "One", "http://x/1.pdf")
    other = Candidate("stub", "2", "Two", "http://x/2.pdf")
    captured: dict = {}
    monkeypatch.setitem(base._REGISTRY, "stub", _stub_provider("stub", [dup, dup, other], captured))
    cands = discovery.discover("q", ["stub"])
    assert [c.id for c in cands] == ["1", "2"]  # (provider, id) dedup
    assert captured["client"] is not None  # a session was created and passed through


def test_discover_unknown_provider_raises():
    with pytest.raises(discovery.DiscoveryError):
        discovery.discover("q", ["does-not-exist"])


# --- manifest emission -----------------------------------------------------------------------------


def test_write_manifest_merges_and_is_acquire_shaped(tmp_path):
    path = tmp_path / "acquire-manifest.yaml"
    first = [Candidate("arxiv", "1", "One", "http://x/1.pdf")]
    added, total = discovery.write_manifest(first, path)
    assert (added, total) == (1, 1)

    # re-run with an overlap + a new one: only the new filename is appended
    second = [Candidate("arxiv", "1", "One", "http://x/1.pdf"), Candidate("arxiv", "2", "Two", "http://x/2.pdf")]
    added, total = discovery.write_manifest(second, path)
    assert (added, total) == (1, 2)

    entries = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert all(set(e) == {"filename", "title", "url"} for e in entries)
    assert [e["filename"] for e in entries] == ["arxiv-1.pdf", "arxiv-2.pdf"]


# --- incremental refresh ---------------------------------------------------------------------------


def _settings(tmp_path) -> Settings:
    return Settings(distilled_dir=tmp_path / "distilled", base_dir=tmp_path)


def test_refresh_persists_and_reuses_last_run(monkeypatch, tmp_path):
    settings = _settings(tmp_path)
    captured: dict = {}
    monkeypatch.setitem(
        base._REGISTRY, "stub", _stub_provider("stub", [Candidate("stub", "1", "One", "http://x/1.pdf")], captured)
    )

    # first run: no persisted state → since is None; marker advances to the injected 'today'
    discovery.refresh("q", ["stub"], settings, today=date(2024, 1, 10))
    assert captured["since"] is None
    assert settings.discovery_state_path.exists()

    # second run: the provider now receives the first run's date as its since
    discovery.refresh("q", ["stub"], settings, today=date(2024, 2, 1))
    assert captured["since"] == date(2024, 1, 10)


def test_last_run_state_round_trip(tmp_path):
    from research_kb.discovery import state

    settings = _settings(tmp_path)
    assert state.load_last_run(settings, "arxiv") is None
    state.save_last_run(settings, "arxiv", date(2024, 5, 1))
    state.save_last_run(settings, "crossref", date(2024, 6, 1))
    assert state.load_last_run(settings, "arxiv") == date(2024, 5, 1)  # preserved across the second save
    assert state.load_last_run(settings, "crossref") == date(2024, 6, 1)
