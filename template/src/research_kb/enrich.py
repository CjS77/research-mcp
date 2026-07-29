"""Enrichment: understanding layered *on top of* verbatim, never replacing it.

Produces derived metadata (named facets + year), one-line section summaries used to contextualize
embeddings, and an optional notation note. When ``ANTHROPIC_API_KEY`` is present the summaries and
metadata come from the LLM working over the verbatim text; offline, a transparent heuristic fills the
same fields so the retrieval layer still benefits. Citation edges are handled in ``citations.py``.
Enrichment output is always stored as ``derived`` content.

Facets are declared by the profile as an ordered list of named vocabularies (``config.FacetSpec`` /
``Settings.facets``); enrichment tags each document with the facet values it exhibits.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .extract.base import ExtractedDoc

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
_HEADING_RE = re.compile(r"^(#{1,4})\s+(.*\S)\s*$")


class SectionSummary(BaseModel):
    section_number: str | None
    section_title: str
    summary: str


class EnrichmentResult(BaseModel):
    """Derived metadata + per-section summaries. Never overwrites verbatim spans."""

    facets: dict[str, list[str]] = Field(default_factory=dict)  # {facet_name: [values]}
    year: int | None = None
    section_summaries: list[SectionSummary] = Field(default_factory=list)
    notation_note: str | None = None
    source: str = "heuristic"  # 'heuristic' | 'llm'

    def summary_for(self, section_title: str | None) -> str | None:
        if not section_title:
            return None
        for s in self.section_summaries:
            if s.section_title == section_title:
                return s.summary
        return None


def _scan_vocab(text: str, vocab: Sequence[str]) -> list[str]:
    found = [t for t in vocab if re.search(rf"(?<![A-Za-z]){re.escape(t)}(?![A-Za-z])", text)]
    # Drop substrings already covered by a longer matched term (e.g. a term nested inside another).
    return [t for t in found if not any(t != o and t in o for o in found)]


def _scan_facets(text: str, settings: Settings) -> dict[str, list[str]]:
    """Tag ``text`` against each declared facet's vocabulary; keep only facets that matched."""
    tagged = {f.name: _scan_vocab(text, f.terms) for f in settings.facets}
    return {name: values for name, values in tagged.items() if values}


def _first_sentence(text: str, limit: int = 200) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    first = _SENT_RE.split(text)[0]
    return (first[:limit] + "…") if len(first) > limit else first


def _section_bodies(md: str) -> list[tuple[str | None, str, str]]:
    """Yield (section_number, section_title, body_text) for each heading in document order."""
    lines = md.splitlines()
    sections: list[tuple[str | None, str, list[str]]] = []
    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            title = m.group(2)
            num_match = re.match(r"^(\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)\.?\s+(.*)$", title)
            number = num_match.group(1) if num_match else None
            clean_title = num_match.group(2) if num_match else title
            sections.append((number, clean_title, []))
        elif sections:
            sections[-1][2].append(line)
    return [(n, t, "\n".join(body)) for n, t, body in sections]


def heuristic_enrich(extracted: ExtractedDoc, title: str, settings: Settings | None = None) -> EnrichmentResult:
    """Offline enrichment: vocab scan for metadata, first-sentence section summaries."""
    settings = settings or get_settings()
    text = extracted.text
    year_match = re.search(r"\b(19|20)\d{2}\b", text[:4000])
    summaries = [
        SectionSummary(section_number=num, section_title=ttl, summary=_first_sentence(body) or ttl)
        for num, ttl, body in _section_bodies(text)
        if ttl
    ]
    return EnrichmentResult(
        facets=_scan_facets(text, settings),
        year=int(year_match.group(0)) if year_match else None,
        section_summaries=summaries,
        notation_note=None,
        source="heuristic",
    )


def enrich(extracted: ExtractedDoc, title: str, settings: Settings | None = None) -> EnrichmentResult:
    """Enrich a document. Uses the LLM path when available, else the heuristic fallback."""
    settings = settings or get_settings()
    if settings.has_anthropic:
        llm_result = _llm_enrich(extracted, title, settings)
        if llm_result is not None:
            return llm_result
    return heuristic_enrich(extracted, title, settings)


def _llm_enrich(extracted: ExtractedDoc, title: str, settings: Settings) -> EnrichmentResult | None:
    """LLM enrichment over the verbatim text. Guarded; returns None to fall back to heuristic."""
    try:
        import json

        import anthropic
    except ImportError:
        return None
    try:
        client = anthropic.Anthropic()
        declared = [f.name for f in settings.facets]
        facet_names = ", ".join(declared) if declared else "(none)"
        prompt = (
            "From the document text below, extract JSON with keys: facets (an object mapping each of "
            f"these facet names to a list of matching terms found in the text: {facet_names} — omit a "
            "name if it has no matches, and use only these names), year (int or null), and "
            "section_summaries (list of {section_number, section_title, summary} — one terse line per "
            "section). Output only JSON.\n\n"
            f"TITLE: {title}\n\n{extracted.text[:120000]}"
        )
        message = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=settings.llm_max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(b.text for b in message.content if getattr(b, "type", None) == "text")
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match is None:
            return None
        data = json.loads(match.group(0))
        return EnrichmentResult(
            facets=_coerce_facets(data.get("facets", {}), declared),
            year=data.get("year"),
            section_summaries=[SectionSummary(**s) for s in data.get("section_summaries", [])],
            notation_note=None,
            source="llm",
        )
    except Exception:
        return None


def _coerce_facets(raw: object, declared: Sequence[str]) -> dict[str, list[str]]:
    """Keep only declared facet names (drop hallucinated keys) with non-empty string-value lists."""
    if not isinstance(raw, dict):
        return {}
    result: dict[str, list[str]] = {}
    for name in declared:
        values = raw.get(name)
        if isinstance(values, list) and values:
            result[name] = [str(v) for v in values]
    return result
