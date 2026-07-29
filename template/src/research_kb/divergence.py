"""Divergence detection: the cross-check between the two independent extractors.

Two extractors disagreeing on an equation or a strong claim is the high-signal event we want,
concentrated on the passages where fidelity matters. This module extracts equations and
candidate claim sentences from both transcriptions, aligns them, and classifies each mismatch as
``cosmetic | notation-only | semantic``. A ``semantic`` divergence blocks indexing of a core
document until a human resolves it. Classification here is a transparent heuristic; an optional
LLM judge (:func:`llm_judge`) can refine it when a key is available.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Literal

from pydantic import BaseModel

Severity = Literal["cosmetic", "notation-only", "semantic"]

_DISPLAY_EQ_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_INLINE_EQ_RE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", re.DOTALL)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")

# Words that make a sentence a "claim" worth cross-checking: negation, strong universals, and
# causal/inferential language carry the meaning most likely to be corrupted by an extractor.
_CLAIM_MARKERS = re.compile(
    r"\b(not|no|never|cannot|can't|must|only|every|all|any|none|always|"
    r"prove[sd]?|show[sn]?|impl(?:y|ies)|require[sd]?|guarantee[sd]?|ensure[sd]?|"
    r"significant|increase[sd]?|decrease[sd]?|cause[sd]?|because)\w*", re.IGNORECASE)
# Notation-only differences: multiplication dot/cross, product spacing, grouping.
_NOTATION_STRIP_RE = re.compile(r"\\cdot|\\times|·|\*|\\,|\\;|\\!|\\ |\\left|\\right|[{}]")
_SIGN_TOKENS = re.compile(r"[+\-]|\\neq|≠|¬|\\ge|\\le|≥|≤|>|<")

# Grammatical words that essentially never occur inside a real equation. The deterministic extractor
# mangles inline math (`x ←$ S`) by sweeping surrounding prose into a `$...$` span; such "equations" carry several of
# these and are dropped before comparison so they cannot manufacture phantom divergences.
_PROSE_WORDS = frozenset(
    "the that this is are was were of from and with which where when while for to be we our can then "
    "thus such denote denotes into over under between their them they it its also because each any all "
    "set element uniformly randomly selected sampling following every so if let there here".split()
)
# Stopwords excluded from claim token-recall (content words carry the meaning we cross-check).
_CLAIM_STOPWORDS = frozenset(
    "the a an of to in on at by for and or is are be as it its this that these those with from we our "
    "can will may such then thus so if let there here into over under between which where when while "
    "each any all not no".split()
)


# A real equation carries at least one of these; a plain sentence fragment carries none.
_MATH_SIGNAL_RE = re.compile(r"[\\=+^_{}<>≤≥≠∈∉·×÷∏∑√∀∃⟨⟩]|\d")


def _nfkd(text: str) -> str:
    """Fold typographic ligatures (ﬁ, ﬂ, …) and compatibility forms to ASCII so the deterministic extractor's `beneﬁcial`
    and the LLM's `beneficial` tokenize identically."""
    return unicodedata.normalize("NFKD", text)


def _looks_like_prose(candidate: str) -> bool:
    """True when a ``$...$`` span is really swept-in prose rather than an equation (the deterministic extractor artifact)."""
    words = set(re.findall(r"[a-z]{2,}", _nfkd(candidate).lower()))
    return sum(1 for w in words if w in _PROSE_WORDS) >= 2


def _is_equation_candidate(candidate: str) -> bool:
    """Keep only spans that look like math: they carry a math signal and are not prose."""
    return bool(_MATH_SIGNAL_RE.search(candidate)) and not _looks_like_prose(candidate)


def _is_math_dense(sentence: str) -> bool:
    """True when a sentence is too math-heavy to compare as prose (the two extractors format math
    incomparably: the deterministic extractor mashes subscripts like ``ai0`` where the LLM emits clean LaTeX)."""
    tokens = re.findall(r"\S+", sentence)
    if not tokens:
        return True
    mathy = sum(1 for t in tokens if _MATH_SIGNAL_RE.search(t) or "$" in t)
    return mathy / len(tokens) > 0.2


def _content_tokens(text: str) -> set[str]:
    """Distinct content words (len ≥ 3, non-stopword, ligature-folded) that carry a claim's meaning."""
    return {w for w in re.findall(r"[a-z0-9]+", _nfkd(text).lower()) if len(w) >= 3 and w not in _CLAIM_STOPWORDS}


# LaTeX wrapper commands whose *names* would otherwise wedge between a symbol and its subscript
# (``\mathsf{H}_{\text{reg}}`` -> ``...mathsf h text reg...``, hiding ``Hreg``). Their names are dropped
# before folding; their arguments are kept.
_LATEX_CMD_RE = re.compile(
    r"\\(?:mathsf|mathcal|mathbb|mathrm|mathbf|mathit|mathfrak|mathscr|boldsymbol|text|textsf|textrm|"
    r"textit|textbf|operatorname|tilde|widetilde|hat|widehat|bar|overline|vec|dot|ddot|mathtt)\b"
)


def _presence_blob(text: str) -> str:
    """Fold text to bare alphanumerics so formatting can't hide content: the deterministic extractor's mashed ``Hreg`` and
    the LLM's ``\\mathsf{H}_{\\text{reg}}`` both collapse to ``hreg``, and a content token is "present"
    iff it is a substring here. Sees through subscript/LaTeX/ligature differences that defeat equality."""
    return re.sub(r"[^a-z0-9]+", "", _LATEX_CMD_RE.sub("", _nfkd(text)).lower())


def _strip_notation(s: str) -> str:
    """Canonicalize presentational multiplication so ``k.G`` and ``k \\cdot G`` compare equal."""
    s = _NOTATION_STRIP_RE.sub("", s)
    return re.sub(r"(?<=[A-Za-z0-9])\.(?=[A-Za-z0-9])", "", s)  # scalar-multiplication dot


class Divergence(BaseModel):
    """A single flagged disagreement between the two extractors."""

    kind: Literal["equation", "claim"]
    severity: Severity
    verbatim: str | None
    llm: str | None
    detail: str


class DivergenceReport(BaseModel):
    """The full cross-check result for one document."""

    source_path: str
    divergences: list[Divergence]

    @property
    def has_blocking(self) -> bool:
        """True if any semantic divergence must be resolved before indexing (core tier)."""
        return any(d.severity == "semantic" for d in self.divergences)

    def counts(self) -> dict[str, int]:
        out = {"cosmetic": 0, "notation-only": 0, "semantic": 0}
        for d in self.divergences:
            out[d.severity] += 1
        return out


def extract_equations(md: str) -> list[str]:
    """Pull display + inline LaTeX from a transcription (deduped, order-preserving)."""
    found = [m.strip() for m in _DISPLAY_EQ_RE.findall(md)]
    masked = _DISPLAY_EQ_RE.sub(" ", md)
    found += [m.strip() for m in _INLINE_EQ_RE.findall(masked)]
    seen: set[str] = set()
    out: list[str] = []
    for e in found:
        key = _norm_eq(e)
        if key and key not in seen and _is_equation_candidate(e):
            seen.add(key)
            out.append(e)
    return out


def extract_claims(md: str) -> list[str]:
    """Candidate claim sentences: prose sentences carrying negation, universal, or causal markers.

    Math-dense sentences are excluded — the two extractors format math incomparably, so a token-level
    comparison of them yields only noise.
    """
    text = re.sub(r"\s+", " ", _DISPLAY_EQ_RE.sub(" ", md))
    sentences = _SENT_SPLIT_RE.split(text)
    return [
        s.strip()
        for s in sentences
        if 20 <= len(s.strip()) <= 400 and _CLAIM_MARKERS.search(s) and not _is_math_dense(s)
    ]


def _norm_eq(eq: str) -> str:
    return re.sub(r"\s+", "", eq.strip().strip("$"))


def _norm_claim(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _best_match(item: str, candidates: list[str], norm) -> tuple[int, float]:
    """Index and ratio of the best-matching candidate under normalization ``norm``."""
    target = norm(item)
    best_i, best_r = -1, 0.0
    for i, c in enumerate(candidates):
        r = SequenceMatcher(None, target, norm(c)).ratio()
        if r > best_r:
            best_i, best_r = i, r
    return best_i, best_r


def _classify_equation(a: str, b: str) -> Severity:
    na, nb = _norm_eq(a), _norm_eq(b)
    if na == nb:
        return "cosmetic"
    # If they match after canonicalizing notation, the difference is presentational.
    if _strip_notation(na) == _strip_notation(nb):
        return "notation-only"
    # A change in the sign/relation/negation skeleton is semantic.
    if _SIGN_TOKENS.findall(na) != _SIGN_TOKENS.findall(nb):
        return "semantic"
    # Different alphanumeric symbol content (variables, subscripts) is semantic.
    if re.sub(r"[^A-Za-z0-9]", "", _strip_notation(na)) != re.sub(r"[^A-Za-z0-9]", "", _strip_notation(nb)):
        return "semantic"
    return "notation-only"


def _claim_severity(sentence: str) -> Severity:
    """A sentence carrying negation or a strong universal/causal claim is ``semantic``; else cosmetic."""
    if re.search(
        r"\b(not|no|never|cannot|can't|only|every|all|must|prove[sd]?|show[sn]?|cause[sd]?|because)\b",
        sentence, re.I,
    ):
        return "semantic"
    return "cosmetic"


def detect_divergences(source_path: str, verbatim_md: str, llm_md: str) -> DivergenceReport:
    """Align equations and claims from the two extractors and flag disagreements."""
    divergences: list[Divergence] = []

    v_eqs, l_eqs = extract_equations(verbatim_md), extract_equations(llm_md)
    matched_l: set[int] = set()
    for eq in v_eqs:
        j, ratio = _best_match(eq, l_eqs, _norm_eq)
        if ratio >= 0.98:
            matched_l.add(j)
        elif ratio >= 0.55:
            matched_l.add(j)
            sev = _classify_equation(eq, l_eqs[j])
            if sev != "cosmetic":
                divergences.append(
                    Divergence(kind="equation", severity=sev, verbatim=eq, llm=l_eqs[j],
                               detail=f"equation match ratio {ratio:.2f}")
                )
        else:
            # Present in only one extractor is a coverage artifact, not a content disagreement — and
            # The deterministic extractor is the unreliable math extractor, so this is almost always its own layout noise.
            # Report it, but do not block on it (only a matched-but-differing equation is semantic).
            divergences.append(
                Divergence(kind="equation", severity="notation-only", verbatim=eq, llm=None,
                           detail="equation present in deterministic extraction only (likely the deterministic extractor layout artifact)")
            )
    for j, eq in enumerate(l_eqs):
        if j not in matched_l:
            divergences.append(
                Divergence(kind="equation", severity="notation-only", verbatim=None, llm=eq,
                           detail="equation present in LLM extraction only")
            )

    # A claim is a divergence only if its *content* is largely absent from the LLM extraction. Each
    # content token is checked for presence as a substring of the folded LLM blob, so segmentation,
    # subscript/LaTeX and ligature differences (the whole source of the earlier noise) don't count as
    # missing content — only genuinely dropped or altered claims do.
    llm_blob = _presence_blob(llm_md)
    for claim in extract_claims(verbatim_md):
        claim_tokens = _content_tokens(claim)
        if len(claim_tokens) < 3:
            continue  # too little content to judge presence reliably
        recall = sum(1 for t in claim_tokens if t in llm_blob) / len(claim_tokens)
        if recall < 0.8:
            divergences.append(
                Divergence(kind="claim", severity=_claim_severity(claim), verbatim=claim, llm=None,
                           detail=f"claim content largely absent from LLM extraction (content recall {recall:.2f})")
            )
    return DivergenceReport(source_path=source_path, divergences=divergences)


def render_report(report: DivergenceReport) -> str:
    """Render ``divergence-report.md``."""
    counts = report.counts()
    lines = [
        f"# Divergence report — {report.source_path}",
        "",
        f"- semantic: **{counts['semantic']}** (blocking)"
        if counts["semantic"]
        else "- semantic: **0**",
        f"- notation-only: {counts['notation-only']}",
        f"- cosmetic: {counts['cosmetic']}",
        "",
        "> Semantic divergences block indexing of a core document until resolved.",
        "",
    ]
    for sev in ("semantic", "notation-only", "cosmetic"):
        items = [d for d in report.divergences if d.severity == sev]
        if not items:
            continue
        lines.append(f"## {sev} ({len(items)})")
        lines.append("")
        for d in items:
            lines.append(f"- **[{d.kind}]** {d.detail}")
            if d.verbatim:
                lines.append(f"  - deterministic: `{d.verbatim[:200]}`")
            if d.llm:
                lines.append(f"  - llm: `{d.llm[:200]}`")
        lines.append("")
    return "\n".join(lines)
