"""Draft a domain *profile* from a topic description — the ``research-kb profile-init`` step.

The engine is corpus-agnostic; a handful of knobs encode domain knowledge (see ``AGENTS.md`` →
"What must be domain-tuned"). At bootstrap those knobs must be authored for the topic — which an LLM
can do well, because it already knows the field's vocabulary. This module turns a one-line topic
description into a **draft** covering every tunable knob:

- **facets** — the named filter axes (``config.FacetSpec`` / ``DEFAULT_FACETS``);
- **atomic-unit keywords** — the field's indivisible units (``chunk._ATOMIC_KEYWORDS``);
- **claim markers** — the words that mark a load-bearing claim (``divergence._CLAIM_MARKERS``);
- **extraction prompt** — the transcription fidelity emphasis (``extract.llm._EXTRACTION_PROMPT``);
- **notation note** — domain notation guidance (``enrich`` / ``EnrichmentResult.notation_note``);

plus guidance for the knobs that need the *actual corpus*, not just the topic (core sources, the
embedding model, the doc-type taxonomy).

**It only proposes.** Nothing here edits ``config.py``/``chunk.py``/… — the draft is rendered to an
editable Markdown file (``work/profile-draft.md``) with paste-ready Python snippets, for a human (or
the bootstrapping agent) to review, tune, and copy into the knobs. The values are emitted in the
knobs' native shapes so they paste straight in.

The LLM path reuses the configured distill backend's optional ``complete`` capability
(:mod:`research_kb.extract.backends`). With no backend available it degrades to a well-commented
**scaffold** — the engine's serving/indexing-offline invariant applies here too: never hard-fail for
lack of a model.
"""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field

from .advisor import GENERAL_EN
from .chunk import _ATOMIC_KEYWORDS
from .config import FacetSpec, Settings, get_settings
from .extract.backends import get_backend
from .extract.llm import _EXTRACTION_PROMPT

# Atomic units bucket into the two ChunkTypes the chunker keeps whole: formal statements
# (theorem-like) and named procedures (algorithm/protocol-like). Mirrors chunk._ATOMIC_KEYWORDS.
AtomicBucket = Literal["theorem", "protocol"]

# Generic default claim markers — a starting point for the scaffold, mirroring the engine's
# divergence._CLAIM_MARKERS default. Plain words a domain expert can edit; the destination regex adds
# the inflection handling (`prove[sd]?`, …), so the draft keeps the vocabulary human-editable.
_DEFAULT_CLAIM_MARKERS: tuple[str, ...] = (
    "not", "no", "never", "cannot", "must", "only", "every", "all", "any", "none", "always",
    "prove", "show", "imply", "require", "guarantee", "ensure",
    "significant", "increase", "decrease", "cause", "because",
)

_CORE_SOURCES_NOTE = (
    "Needs the actual corpus, not just the topic — fill after discovery. List the load-bearing works "
    "an agent may quote directly (matched against the source-file stem)."
)
_EMBED_NOTE = (
    "The default is general English. Run `research-kb advise` once the corpus is seeded to get a "
    "language/density-based recommendation, and confirm it with `research-kb eval --ab`."
)
_DOC_TYPES_NOTE = (
    "The default taxonomy is paper/research/assessment/sketch/spec (models.DocType). Rename to the "
    "document kinds this topic actually has (e.g. clause/holding, trial/cohort, listing/figure)."
)


class AtomicUnit(BaseModel):
    """One indivisible-unit keyword and the ChunkType bucket it maps to (statement vs procedure)."""

    keyword: str
    bucket: AtomicBucket


class ProfileDraft(BaseModel):
    """A proposed profile covering every tunable knob. Never applied automatically; rendered for review."""

    topic: str
    source: Literal["llm", "scaffold"]
    facets: list[FacetSpec] = Field(default_factory=list)
    atomic_units: list[AtomicUnit] = Field(default_factory=list)
    claim_markers: list[str] = Field(default_factory=list)
    extraction_prompt: str = ""
    notation_note: str = ""  # empty string = no special notation (prose fields)
    core_sources_note: str = _CORE_SOURCES_NOTE
    embed_model: str = GENERAL_EN.model
    embed_note: str = _EMBED_NOTE
    doc_types_note: str = _DOC_TYPES_NOTE


# --- Drafting ------------------------------------------------------------------------------------


def draft_profile(topic: str, settings: Settings | None = None, *, offline: bool = False) -> ProfileDraft:
    """Draft a profile for ``topic``: LLM-authored when a backend is available, else the scaffold.

    ``offline`` forces the scaffold even when a backend could run. The LLM path never fails the call —
    a missing backend, an empty response, or unparseable output all fall back to the scaffold, and any
    knob the model omits is filled from the scaffold so the draft always covers every knob.
    """
    settings = settings or get_settings()
    scaffold = scaffold_draft(topic)
    if offline:
        return scaffold
    backend = get_backend(settings.llm_extract_backend)
    if backend is None or backend.complete is None or not backend.available(settings):
        return scaffold
    raw = backend.complete(build_prompt(topic), settings)
    if not raw:
        return scaffold
    return parse_llm_response(raw, topic, fallback=scaffold) or scaffold


def scaffold_draft(topic: str) -> ProfileDraft:
    """The offline draft: placeholder facets to fill in, plus the engine's generic defaults elsewhere.

    Concrete and paste-ready so a KB is queryable immediately, while flagging (via the facet
    placeholders and the guidance notes) what a human or the bootstrapping agent should tune.
    """
    atomic_units = [AtomicUnit(keyword=kw, bucket=_as_bucket(ctype)) for kw, ctype in _ATOMIC_KEYWORDS.items()]
    return ProfileDraft(
        topic=topic,
        source="scaffold",
        facets=[
            FacetSpec(name="<facet-1-name>", terms=("<term>", "<term>")),
            FacetSpec(name="<facet-2-name>", terms=("<term>", "<term>")),
        ],
        atomic_units=atomic_units,
        claim_markers=list(_DEFAULT_CLAIM_MARKERS),
        extraction_prompt=_EXTRACTION_PROMPT,
        notation_note="",
    )


def build_prompt(topic: str) -> str:
    """The instruction sent to the LLM: draft the domain knobs for ``topic`` as strict JSON."""
    return (
        "You are configuring a fidelity-first retrieval knowledge base for the topic below. Using your "
        "knowledge of this field's vocabulary, propose values for the engine's domain knobs. Return "
        "ONLY a single JSON object (no prose, no code fences) with these keys:\n"
        '  "facets": a list of 2-4 objects {"name": <short filter-axis name, snake_case>, "terms": '
        "[<controlled-vocabulary values on that axis>]}. Facets are the most useful axes to narrow a "
        "search by in this field (e.g. clade/period for dinosaurs, compound/condition for a drug field).\n"
        '  "atomic_units": a list of {"keyword": <word that opens an indivisible unit>, "bucket": '
        '"theorem" | "protocol"}. "theorem" = a formal statement kept whole (theorem/lemma/clause/'
        'holding/trial…); "protocol" = a named procedure (algorithm/protocol/procedure/recipe…).\n'
        '  "claim_markers": a list of lowercase words that mark a load-bearing claim in this field '
        "(negations, strong universals, causal/inferential verbs).\n"
        '  "extraction_prompt": a faithful-transcription instruction tuned for this medium (emphasise '
        "equations for STEM, statute/section structure for law, dosages/tables for medicine, etc.). It "
        "must forbid summarising or rewording and ask for Markdown output.\n"
        '  "notation_note": one sentence of domain notation guidance, or "" for prose fields with none.\n'
        '  "embed_model": a suggested embedding model name (default "' + GENERAL_EN.model + '" for '
        "general English; prefer a scientific or multilingual model where the corpus warrants).\n\n"
        f"TOPIC: {topic}\n"
    )


def parse_llm_response(raw: str, topic: str, *, fallback: ProfileDraft) -> ProfileDraft | None:
    """Parse the model's JSON into a draft; ``None`` if no JSON is recoverable (caller uses scaffold).

    Missing or malformed knobs fall back to ``fallback`` (the scaffold) so a partial response still
    yields a complete draft. Only a wholesale parse failure returns ``None``.
    """
    data = _extract_json(raw)
    if data is None:
        return None
    facets = _coerce_facets(data.get("facets")) or fallback.facets
    atomic = _coerce_atomic(data.get("atomic_units")) or fallback.atomic_units
    markers = _coerce_str_list(data.get("claim_markers")) or fallback.claim_markers
    prompt = _coerce_str(data.get("extraction_prompt")) or fallback.extraction_prompt
    notation = _coerce_str(data.get("notation_note"))
    embed_model = _coerce_str(data.get("embed_model")) or fallback.embed_model
    return ProfileDraft(
        topic=topic,
        source="llm",
        facets=facets,
        atomic_units=atomic,
        claim_markers=markers,
        extraction_prompt=prompt,
        notation_note=notation if notation is not None else fallback.notation_note,
        embed_model=embed_model,
    )


# --- Coercion helpers (the model's JSON is untrusted; keep only well-formed values) ---------------


def _extract_json(raw: str) -> dict[str, object] | None:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _coerce_facets(raw: object) -> list[FacetSpec]:
    if not isinstance(raw, list):
        return []
    out: list[FacetSpec] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        terms_raw = item.get("terms")
        terms = terms_raw if isinstance(terms_raw, list) else []
        clean = tuple(str(t).strip() for t in terms if str(t).strip())
        out.append(FacetSpec(name=name.strip(), terms=clean))
    return out


def _coerce_atomic(raw: object) -> list[AtomicUnit]:
    if not isinstance(raw, list):
        return []
    out: list[AtomicUnit] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        kw = item.get("keyword")
        if not isinstance(kw, str) or not kw.strip():
            continue
        out.append(AtomicUnit(keyword=kw.strip().lower(), bucket=_as_bucket(item.get("bucket"))))
    return out


def _coerce_str_list(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(x).strip().lower() for x in raw if str(x).strip()]


def _coerce_str(raw: object) -> str | None:
    return raw.strip() if isinstance(raw, str) else None


def _as_bucket(value: object) -> AtomicBucket:
    """Narrow an arbitrary value to a valid atomic bucket (default ``theorem``)."""
    return "protocol" if value == "protocol" else "theorem"


# --- Rendering: the editable Markdown draft ------------------------------------------------------


def render_draft(draft: ProfileDraft) -> str:
    """Render the draft as a Markdown guide: one section per knob, each with a paste-ready snippet."""
    facet_names = ", ".join(f.name for f in draft.facets) or "(none yet — fill in)"
    parts = [
        f"# Draft domain profile — {draft.topic}",
        "",
        f"Generated by `research-kb profile-init` (source: **{draft.source}**). "
        + (
            "An LLM proposed these values from the topic description."
            if draft.source == "llm"
            else "No LLM backend was available, so this is a scaffold with the engine's generic "
            "defaults and placeholders — fill it in (a bootstrapping agent that knows the field can too)."
        ),
        "",
        "> **This is a proposal, not a config.** Nothing here is applied automatically. Review and edit,"
        " then copy each block into the file named in its heading. See `AGENTS.md` →"
        ' "What must be domain-tuned" for the full knob reference.',
        "",
        f"Proposed facets: {facet_names}",
        "",
        "---",
        "",
        _render_facets_section(draft),
        _render_atomic_section(draft),
        _render_claim_markers_section(draft),
        _render_extraction_prompt_section(draft),
        _render_notation_section(draft),
        _render_corpus_dependent_section(draft),
    ]
    return "\n".join(parts).rstrip() + "\n"


def _py_block(code: str) -> str:
    return "```python\n" + code + "\n```"


def _render_facets_section(draft: ProfileDraft) -> str:
    lines = ["DEFAULT_FACETS: tuple[FacetSpec, ...] = ("]
    if draft.facets:
        for f in draft.facets:
            terms = ", ".join(f'"{t}"' for t in f.terms)
            trailing = "," if len(f.terms) == 1 else ""
            lines.append(f'    FacetSpec(name="{f.name}", terms=({terms}{trailing})),')
    else:
        lines.append("    # TODO: declare this topic's filter axes")
    lines.append(")")
    return (
        "## 1. Facets — `config.py` `DEFAULT_FACETS`\n\n"
        "The named axes an agent narrows a search by (`kb_search(..., facets={name: value})`). Pick the "
        "2-4 most useful, and list the controlled vocabulary that tags a document on each. The facet "
        "*names* also become the query parameter keys.\n\n"
        + _py_block("\n".join(lines))
        + "\n"
    )


def _render_atomic_section(draft: ProfileDraft) -> str:
    lines = ["_ATOMIC_KEYWORDS: dict[str, ChunkType] = {"]
    for bucket in ("theorem", "protocol"):
        kws = [u.keyword for u in draft.atomic_units if u.bucket == bucket]
        if kws:
            comment = "formal statements (kept whole)" if bucket == "theorem" else "named procedures"
            entries = ", ".join(f'"{kw}": "{bucket}"' for kw in kws)
            lines.append(f"    {entries},  # {comment}")
    lines.append("}")
    return (
        "## 2. Atomic-unit keywords — `chunk.py` `_ATOMIC_KEYWORDS`\n\n"
        "The field's indivisible units — the chunker never splits these even when oversized. Two "
        'buckets: `"theorem"` (a formal statement kept whole) and `"protocol"` (a named procedure).\n\n'
        + _py_block("\n".join(lines))
        + "\n"
    )


def _render_claim_markers_section(draft: ProfileDraft) -> str:
    body = "|".join(draft.claim_markers)
    code = (
        "_CLAIM_MARKERS = re.compile(\n"
        f'    r"\\b({body})\\w*",\n'
        "    re.IGNORECASE,\n"
        ")"
    )
    return (
        "## 3. Claim markers — `divergence.py` `_CLAIM_MARKERS`\n\n"
        "The words that make a sentence a load-bearing *claim* worth cross-checking between the two "
        "extractors — negations, strong universals, causal/inferential verbs. Add inflections in the "
        "regex where useful (e.g. `prove[sd]?`).\n\n"
        + _py_block(code)
        + "\n"
    )


def _render_extraction_prompt_section(draft: ProfileDraft) -> str:
    # ensure_ascii=False keeps em-dashes/accents literal in the paste-ready snippet (still valid Python).
    code = "_EXTRACTION_PROMPT = " + json.dumps(draft.extraction_prompt, ensure_ascii=False)
    return (
        "## 4. Extraction prompt — `extract/llm.py` `_EXTRACTION_PROMPT`\n\n"
        "The fidelity instruction for the independent LLM transcription. Emphasise what must be "
        "preserved exactly in this medium (equations for STEM, statute structure for law, tables and "
        "dosages for medicine, …); it must forbid summarising or rewording.\n\n"
        + _py_block(code)
        + "\n"
    )


def _render_notation_section(draft: ProfileDraft) -> str:
    if draft.notation_note:
        code = f"notation_note = {json.dumps(draft.notation_note, ensure_ascii=False)}"
    else:
        code = "notation_note = None  # prose field — no special notation guidance"
    return (
        "## 5. Notation note — `enrich.py` (`EnrichmentResult.notation_note`)\n\n"
        "Domain notation guidance carried onto enriched chunks (currently `None` by default). Set it in "
        "`heuristic_enrich` / `_llm_enrich`, or leave `None` for prose fields.\n\n"
        + _py_block(code)
        + "\n"
    )


def _render_corpus_dependent_section(draft: ProfileDraft) -> str:
    return (
        "## 6. Also tune — needs the actual corpus, not just the topic\n\n"
        f"- **Core sources** — `config.py` `DEFAULT_CORE_SOURCES`. {draft.core_sources_note}\n"
        f"- **Embedding model** — `KB_EMBED_MODEL` (suggested: `{draft.embed_model}`). {draft.embed_note}\n"
        f"- **doc_type taxonomy** — `models.py` `DocType`. {draft.doc_types_note}\n"
        "- **Package / script / server names** — `pyproject.toml`, `mcp_server.py`, `.mcp.json`: "
        "rename `research-kb` to `<name>-kb` (bootstrap step 1, mechanical — not domain vocabulary).\n"
    )
