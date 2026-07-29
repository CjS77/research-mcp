"""``profile-init``: drafting a domain profile from a topic description (LLM mocked — offline).

Every test avoids the network and any real model: the LLM backend is a fake whose ``complete``
returns canned text, and the scaffold path needs no backend at all.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from click.testing import CliRunner

from research_kb import profile
from research_kb.cli import main
from research_kb.config import Settings
from research_kb.extract import backends, claude_cli
from research_kb.extract.backends import Backend
from research_kb.extract.llm import _EXTRACTION_PROMPT

# A well-formed model response for a paleontology KB — the "happy path" the LLM would return.
_CANNED = {
    "facets": [
        {"name": "clade", "terms": ["theropod", "sauropod", "ornithischian"]},
        {"name": "period", "terms": ["triassic", "jurassic", "cretaceous"]},
    ],
    "atomic_units": [
        {"keyword": "specimen", "bucket": "theorem"},
        {"keyword": "excavation", "bucket": "protocol"},
    ],
    "claim_markers": ["not", "only", "earliest", "largest"],
    "extraction_prompt": (
        "Transcribe this paleontology PDF faithfully, preserving specimen tables and stratigraphic "
        "figures. Do not summarize, correct, or reword. Output only Markdown."
    ),
    "notation_note": "Use standard binomial nomenclature (Genus species).",
    "embed_model": "BAAI/bge-base-en-v1.5",
}

# The five knobs every draft must cover, as they appear in the rendered Markdown destinations.
_KNOB_MARKERS = ("DEFAULT_FACETS", "_ATOMIC_KEYWORDS", "_CLAIM_MARKERS", "_EXTRACTION_PROMPT", "notation_note")


def _fake_backend(complete_fn, *, available: bool = True) -> Backend:
    return Backend(name="fake", available=lambda s: available, extract=lambda p, s: None, complete=complete_fn)


def _use_backend(monkeypatch, backend: Backend) -> None:
    monkeypatch.setattr(profile, "get_backend", lambda name: backend)


# --- Scaffold (offline) --------------------------------------------------------------------------


def test_scaffold_covers_every_knob():
    draft = profile.draft_profile("dinosaurs", Settings(), offline=True)
    assert draft.source == "scaffold"
    # Placeholder facets (names + terms), real default atomic units / claim markers / extraction prompt.
    assert [f.name for f in draft.facets] == ["<facet-1-name>", "<facet-2-name>"]
    assert all(f.terms for f in draft.facets)
    assert draft.atomic_units and {u.bucket for u in draft.atomic_units} == {"theorem", "protocol"}
    assert "not" in draft.claim_markers
    assert draft.extraction_prompt == _EXTRACTION_PROMPT

    rendered = profile.render_draft(draft)
    for marker in _KNOB_MARKERS:
        assert marker in rendered
    assert "source: **scaffold**" in rendered
    assert "<facet-1-name>" in rendered  # placeholder survives into the guide
    # Corpus-dependent knobs are surfaced too.
    assert "DEFAULT_CORE_SOURCES" in rendered and "KB_EMBED_MODEL" in rendered and "DocType" in rendered


def test_draft_falls_back_to_scaffold_when_backend_unavailable(monkeypatch):
    # A backend that exists but reports unavailable must not be invoked; we get the scaffold.
    called = {"n": 0}

    def _complete(prompt, s):
        called["n"] += 1
        return json.dumps(_CANNED)

    _use_backend(monkeypatch, _fake_backend(_complete, available=False))
    draft = profile.draft_profile("dinosaurs", Settings())
    assert draft.source == "scaffold"
    assert called["n"] == 0


# --- LLM path ------------------------------------------------------------------------------------


def test_llm_path_parses_and_covers_every_knob(monkeypatch):
    _use_backend(monkeypatch, _fake_backend(lambda prompt, s: json.dumps(_CANNED)))
    draft = profile.draft_profile("dinosaurs", Settings())

    assert draft.source == "llm"
    assert [f.name for f in draft.facets] == ["clade", "period"]
    assert "theropod" in draft.facets[0].terms
    buckets = {u.keyword: u.bucket for u in draft.atomic_units}
    assert buckets == {"specimen": "theorem", "excavation": "protocol"}
    assert "earliest" in draft.claim_markers
    assert "paleontology" in draft.extraction_prompt
    assert "binomial" in draft.notation_note

    rendered = profile.render_draft(draft)
    for marker in _KNOB_MARKERS:
        assert marker in rendered
    assert 'FacetSpec(name="clade"' in rendered
    assert '"specimen": "theorem"' in rendered and '"excavation": "protocol"' in rendered
    assert "earliest" in rendered and "paleontology" in rendered and "binomial" in rendered


def test_llm_prompt_wraps_the_topic(monkeypatch):
    seen = {}

    def _complete(prompt, s):
        seen["prompt"] = prompt
        return json.dumps(_CANNED)

    _use_backend(monkeypatch, _fake_backend(_complete))
    profile.draft_profile("cretaceous dinosaurs", Settings())
    assert "cretaceous dinosaurs" in seen["prompt"]
    assert "facets" in seen["prompt"] and "atomic_units" in seen["prompt"]


def test_llm_garbage_response_falls_back_to_scaffold(monkeypatch):
    _use_backend(monkeypatch, _fake_backend(lambda prompt, s: "sorry, I could not help with that"))
    draft = profile.draft_profile("dinosaurs", Settings())
    assert draft.source == "scaffold"


def test_llm_empty_response_falls_back_to_scaffold(monkeypatch):
    _use_backend(monkeypatch, _fake_backend(lambda prompt, s: None))
    draft = profile.draft_profile("dinosaurs", Settings())
    assert draft.source == "scaffold"


def test_partial_llm_response_filled_from_scaffold(monkeypatch):
    # Only facets returned; the other knobs are backfilled from the scaffold so the draft stays complete.
    partial = {"facets": [{"name": "clade", "terms": ["theropod"]}]}
    _use_backend(monkeypatch, _fake_backend(lambda prompt, s: json.dumps(partial)))
    draft = profile.draft_profile("dinosaurs", Settings())

    assert draft.source == "llm"
    assert [f.name for f in draft.facets] == ["clade"]
    assert draft.atomic_units and "not" in draft.claim_markers  # from scaffold
    assert draft.extraction_prompt == _EXTRACTION_PROMPT


def test_offline_flag_skips_available_backend(monkeypatch):
    _use_backend(monkeypatch, _fake_backend(lambda prompt, s: json.dumps(_CANNED)))
    draft = profile.draft_profile("dinosaurs", Settings(), offline=True)
    assert draft.source == "scaffold"


def test_coerce_atomic_defaults_unknown_bucket_to_theorem():
    units = profile._coerce_atomic([{"keyword": "Clause", "bucket": "nonsense"}, {"keyword": "recipe"}])
    assert units[0].keyword == "clause" and units[0].bucket == "theorem"
    assert units[1].bucket == "theorem"


# --- Backend text-completion capability ----------------------------------------------------------


def test_shipped_backends_expose_complete_where_expected():
    assert backends.get_backend("claude_cli").complete is not None
    assert backends.get_backend("api").complete is not None
    assert backends.get_backend("none").complete is None


def test_build_prompt_command_grants_no_tools_or_bypass():
    cmd = claude_cli.build_prompt_command("draft a profile", Settings(claude_model="sonnet", claude_effort="high"))
    assert cmd[:3] == ["claude", "-p", "draft a profile"]
    assert "--tools" not in cmd  # a text prompt needs no file access …
    assert "--permission-mode" not in cmd  # … so nothing to bypass
    assert "--model" in cmd and "sonnet" in cmd and "--effort" in cmd


def test_run_claude_prompt_returns_stdout(monkeypatch):
    monkeypatch.setattr(claude_cli, "claude_available", lambda settings=None: True)
    monkeypatch.setattr(
        claude_cli.subprocess, "run",
        lambda cmd, **kw: SimpleNamespace(returncode=0, stdout="RESULT\n", stderr=""),
    )
    assert claude_cli.run_claude_prompt("hi", Settings()) == "RESULT"


def test_run_claude_prompt_failure_returns_none(monkeypatch):
    monkeypatch.setattr(claude_cli, "claude_available", lambda settings=None: True)
    monkeypatch.setattr(
        claude_cli.subprocess, "run",
        lambda cmd, **kw: SimpleNamespace(returncode=1, stdout="", stderr="boom"),
    )
    assert claude_cli.run_claude_prompt("hi", Settings()) is None


# --- CLI -----------------------------------------------------------------------------------------


def test_cli_profile_init_writes_editable_draft(tmp_path):
    dest = tmp_path / "profile-draft.md"
    result = CliRunner().invoke(main, ["profile-init", "dinosaurs of the cretaceous", "--offline", "--out", str(dest)])
    assert result.exit_code == 0, result.output
    assert dest.exists()
    text = dest.read_text(encoding="utf-8")
    assert text.startswith("# Draft domain profile — dinosaurs of the cretaceous")
    for marker in _KNOB_MARKERS:
        assert marker in text
    assert "scaffold" in result.output
