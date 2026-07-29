"""Embedding-model advisor: recommend ``KB_EMBED_MODEL`` from cheap, offline corpus signals.

The advisor inspects a text sample of the corpus (indexed chunks, or the ``reference/`` files when the
index is empty) and estimates two axes with explicit, network-free heuristics:

- **Language** — the share of letters written in a non-Latin script, plus the frequency of common
  English function words. A corpus that is mostly non-Latin, or Latin-script but poor in English
  stopwords (e.g. French/German), is routed to a multilingual encoder.
- **Density** — an "academic-ness" score built from citation brackets, formal-claim markers, maths
  notation and mean word length. A dense academic corpus is routed to a larger encoder.

It *recommends*; the gold-query eval (``run_ab_eval``) *decides*. Nothing here downloads a model or
touches the network — it only reads text and returns a name.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Settings, get_settings


@dataclass(frozen=True)
class EmbedModel:
    """A candidate embedding model: its name, vector dimension, and why one would pick it."""

    model: str
    dim: int
    rationale: str


# The small catalogue the advisor chooses among. All three are fastembed-servable and download on
# first index; the DB records whichever is used, so the choice is a rebuild, not a migration.
GENERAL_EN = EmbedModel(
    "BAAI/bge-base-en-v1.5", 768,
    "general English prose — balanced size/quality, the safe default",
)
DENSE_EN = EmbedModel(
    "BAAI/bge-large-en-v1.5", 1024,
    "dense academic English — a larger encoder captures technical/formal text better "
    "(swap in a domain-specific scientific model, e.g. allenai/specter2 or a PubMedBERT, when one fits)",
)
MULTILINGUAL = EmbedModel(
    "BAAI/bge-m3", 1024,
    "non-English or mixed-language corpus — a multilingual encoder keeps cross-lingual queries in one space",
)

CATALOG: tuple[EmbedModel, ...] = (GENERAL_EN, DENSE_EN, MULTILINGUAL)

# Common English function words: their share of tokens is a cheap language signal. English prose sits
# well above 0.15; Latin-script non-English (French, German, …) sits far below it.
_ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    "the of and to in a is that for it as are with be this by on not or an we from at which "
    "was were has have had but their can will they its into such these than then them our".split()
)

# Words that mark a load-bearing, formal claim — dense academic writing is thick with them.
_ACADEMIC_MARKERS: frozenset[str] = frozenset(
    "theorem lemma proof corollary proposition definition hypothesis empirical coefficient "
    "abstract methodology dataset regression variance estimator convergence asymptotic algorithm "
    "figure equation appendix parameters analysis significant conclude respectively".split()
)

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)  # letters only (any script), no digits/underscores
_ASCII_WORD_RE = re.compile(r"[a-z]+")
_CITATION_RE = re.compile(r"\[\d{1,3}\]|\bet\s+al\.?|\bdoi\b|\((?:19|20)\d{2}\)", re.IGNORECASE)
_MATH_RE = re.compile(r"\\[a-zA-Z]+|[=≤≥±∑∏∫√≈∈∀∃]|\$[^$]{1,80}\$|\^\{|_\{")

# Non-Latin script code-point ranges (script, not accents — "café" stays Latin, "Кириллица" does not).
_NON_LATIN_RANGES: tuple[tuple[int, int], ...] = (
    (0x0370, 0x03FF),  # Greek
    (0x0400, 0x04FF),  # Cyrillic
    (0x0530, 0x058F),  # Armenian
    (0x0590, 0x05FF),  # Hebrew
    (0x0600, 0x06FF),  # Arabic
    (0x0900, 0x097F),  # Devanagari
    (0x3040, 0x30FF),  # Hiragana + Katakana
    (0x3400, 0x4DBF),  # CJK Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xAC00, 0xD7A3),  # Hangul syllables
)


@dataclass
class CorpusSignals:
    """The measured, explainable signals the recommendation is derived from."""

    source: str  # "db" | "reference" | "text"
    n_docs: int
    n_chars: int
    n_tokens: int
    non_latin_ratio: float
    english_stopword_ratio: float
    academic_score: float

    def as_dict(self) -> dict[str, object]:
        d = asdict(self)
        for key in ("non_latin_ratio", "english_stopword_ratio", "academic_score"):
            d[key] = round(float(d[key]), 4)
        return d


@dataclass
class EmbedRecommendation:
    """The advisor's verdict: a model plus the signals and one-line rationale behind it."""

    model: str
    dim: int
    rationale: str
    signals: CorpusSignals

    def as_dict(self) -> dict[str, object]:
        return {"model": self.model, "dim": self.dim, "rationale": self.rationale, "signals": self.signals.as_dict()}


def _is_non_latin(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _NON_LATIN_RANGES)


def _academic_score(text: str, tokens: list[str], n_tokens: int, n_chars: int) -> float:
    """A 0–1 density score: citations + formal markers + maths notation + mean word length."""
    if n_tokens == 0:
        return 0.0
    marker_rate = sum(1 for t in tokens if t in _ACADEMIC_MARKERS) / n_tokens
    citation_rate = len(_CITATION_RE.findall(text)) / n_tokens
    math_rate = len(_MATH_RE.findall(text)) / max(n_chars, 1)
    avg_word_len = sum(len(t) for t in tokens) / n_tokens

    score = 0.0
    score += min(marker_rate * 40.0, 0.45)
    score += min(citation_rate * 25.0, 0.30)
    score += min(math_rate * 150.0, 0.20)
    if avg_word_len > 6.0:
        score += min((avg_word_len - 6.0) * 0.10, 0.15)
    return min(score, 1.0)


def analyze_text(text: str, *, source: str = "text", n_docs: int = 1) -> CorpusSignals:
    """Measure language + density signals over a text sample. Pure, cheap, and offline."""
    letters = [ch for ch in text if ch.isalpha()]
    non_latin = sum(1 for ch in letters if _is_non_latin(ch))
    non_latin_ratio = non_latin / len(letters) if letters else 0.0

    tokens = _ASCII_WORD_RE.findall(text.lower())  # ASCII words for the English-stopword signal
    all_tokens = [m.group(0).lower() for m in _WORD_RE.finditer(text)]  # any-script words for length/density
    n_tokens = len(all_tokens)
    stopword_ratio = sum(1 for t in tokens if t in _ENGLISH_STOPWORDS) / len(tokens) if tokens else 0.0

    return CorpusSignals(
        source=source,
        n_docs=n_docs,
        n_chars=len(text),
        n_tokens=n_tokens,
        non_latin_ratio=non_latin_ratio,
        english_stopword_ratio=stopword_ratio,
        academic_score=_academic_score(text, all_tokens, n_tokens, len(text)),
    )


def recommend(signals: CorpusSignals) -> EmbedRecommendation:
    """Map corpus signals to a model. Explicit thresholds — the advisor proposes, the eval disposes."""
    non_english = signals.non_latin_ratio > 0.10 or (signals.n_tokens >= 20 and signals.english_stopword_ratio < 0.08)
    if non_english:
        choice = MULTILINGUAL
    elif signals.academic_score >= 0.35:
        choice = DENSE_EN
    else:
        choice = GENERAL_EN
    return EmbedRecommendation(model=choice.model, dim=choice.dim, rationale=choice.rationale, signals=signals)


def dim_for_model(model: str) -> int | None:
    """The catalogue dimension for a known model name, else None (caller falls back to config)."""
    return next((m.dim for m in CATALOG if m.model == model), None)


def _text_from_db(con: sqlite3.Connection, char_budget: int) -> tuple[str, int]:
    """Sample chunk content from the index. Returns (text, n_docs). Empty when nothing is indexed."""
    rows = con.execute(
        "SELECT content FROM chunks WHERE content_kind = 'verbatim' ORDER BY document_id, chunk_index"
    ).fetchall()
    parts: list[str] = []
    total = 0
    for row in rows:
        chunk = row["content"] or ""
        parts.append(chunk)
        total += len(chunk)
        if total >= char_budget:
            break
    n_docs = con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    return "\n".join(parts), int(n_docs)


def _read_source_text(path: Path, per_doc_chars: int) -> str:
    """A capped text sample from one corpus file (PDF first pages, or head of a text file)."""
    if path.suffix.lower() == ".pdf":
        try:
            import fitz  # pymupdf; lazy so a text-only corpus never imports it

            with fitz.open(path) as doc:
                pages = [doc[i].get_text() for i in range(min(doc.page_count, 3))]
            return "".join(pages)[:per_doc_chars]
        except Exception:
            return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:per_doc_chars]
    except OSError:
        return ""


def _text_from_reference(settings: Settings, char_budget: int, per_doc_chars: int) -> tuple[str, int]:
    """Sample text straight from the ``reference/`` corpus root when the index is empty."""
    root = settings.reference_dir
    if not root.exists():
        return "", 0
    suffixes = {".pdf", ".md", ".markdown", ".txt"}
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)
    parts: list[str] = []
    total = 0
    n_docs = 0
    for path in files:
        sample = _read_source_text(path, per_doc_chars)
        if not sample:
            continue
        n_docs += 1
        parts.append(sample)
        total += len(sample)
        if total >= char_budget:
            break
    return "\n".join(parts), n_docs


def analyze_corpus(
    settings: Settings | None = None,
    con: sqlite3.Connection | None = None,
    *,
    prefer: str = "auto",
    char_budget: int = 400_000,
    per_doc_chars: int = 6_000,
) -> CorpusSignals:
    """Gather signals from the index if it has content, else from ``reference/``.

    ``prefer`` is ``"auto"`` (index when non-empty, else reference), ``"db"`` (index only), or
    ``"reference"`` (files only). Sampling is capped by ``char_budget`` so the pass stays cheap.
    """
    settings = settings or get_settings()
    text, n_docs, source = "", 0, "reference"

    if con is not None and prefer in {"auto", "db"}:
        text, n_docs = _text_from_db(con, char_budget)
        source = "db"

    if (prefer == "reference") or (not text and prefer != "db"):
        text, n_docs = _text_from_reference(settings, char_budget, per_doc_chars)
        source = "reference"

    return analyze_text(text, source=source, n_docs=n_docs)


def advise(
    settings: Settings | None = None,
    con: sqlite3.Connection | None = None,
    *,
    prefer: str = "auto",
) -> EmbedRecommendation:
    """One-shot: analyse the corpus and return a model recommendation."""
    return recommend(analyze_corpus(settings, con, prefer=prefer))
