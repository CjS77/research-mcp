"""Divergence detector: extraction, classification, blocking, rendering."""

from __future__ import annotations

from research_kb.divergence import (
    _classify_equation,
    detect_divergences,
    extract_claims,
    extract_equations,
    render_report,
)


def test_extract_equations_dedup():
    md = "inline $a = b$ and display $$c = d$$ and again $a = b$."
    eqs = extract_equations(md)
    assert any("c = d" in e for e in eqs)
    assert sum(1 for e in eqs if "a = b" in e) == 1  # deduped


def test_extract_claims_filters_to_markers():
    md = "The sky is blue. The method must converge for all valid inputs. Cats purr."
    claims = extract_claims(md)
    assert any("must converge" in c for c in claims)
    assert not any("Cats purr" in c for c in claims)


def test_notation_only_vs_semantic():
    assert _classify_equation("R = k.G", "R = k \\cdot G") == "notation-only"
    assert _classify_equation("t \\le n", "t \\ge n") == "semantic"
    assert _classify_equation("y = g^x", "y = g^z") == "semantic"


def test_semantic_divergence_is_blocking():
    verbatim = "We require $t \\le n$ items and the claim must hold."
    llm = "We require $t \\ge n$ items and the claim cannot hold."
    report = detect_divergences("p.pdf", verbatim, llm)
    assert report.has_blocking
    assert report.counts()["semantic"] >= 1
    text = render_report(report)
    assert "Divergence report" in text and "blocking" in text


def test_identical_extractions_have_no_semantic_divergence():
    md = "The method is efficient. We use $R = a.b$ throughout."
    report = detect_divergences("p.pdf", md, md)
    assert not report.has_blocking


def test_prose_swept_into_a_math_span_is_not_an_equation():
    # pymupdf mangles inline notation by wrapping surrounding prose in a `$...$` span; such a span is
    # prose, not an equation, and must not manufacture a phantom "present in one extractor only" flag.
    md = "$that x is uniformly randomly selected from the set S$"
    assert extract_equations(md) == []


def test_present_claim_with_different_math_formatting_is_not_flagged():
    # The same sentence: the deterministic extractor mashes subscripts (`Rlow`), the LLM uses LaTeX
    # (`\mathsf{R}_{\text{low}}`). Folding must see the content as present in both.
    verbatim = "The system S1 may access Rlow, Rmid, and Rhigh, but not the control channel."
    llm = (
        "The system $\\mathcal{S}_1$ may access $\\mathsf{R}_{\\text{low}}$, "
        "$\\mathsf{R}_{\\text{mid}}$, and $\\mathsf{R}_{\\text{high}}$, but not the control channel."
    )
    assert detect_divergences("p.pdf", verbatim, llm).counts()["semantic"] == 0


def test_ligature_difference_is_not_a_divergence():
    # The deterministic extractor keeps the `ﬁ` ligature; the LLM normalizes to ASCII. NFKD folding must reconcile them.
    verbatim = "This is an eﬃcient and veriﬁable method that works in every case."
    llm = "This is an efficient and verifiable method that works in every case."
    assert detect_divergences("p.pdf", verbatim, llm).counts()["semantic"] == 0


def test_genuinely_dropped_claim_is_flagged():
    verbatim = "The method cannot recover the original signal under the stated assumption."
    llm = "This transcription concerns entirely unrelated bookkeeping matters."
    assert detect_divergences("p.pdf", verbatim, llm).counts()["semantic"] >= 1


def test_single_extractor_equation_is_reported_but_not_blocking():
    verbatim = "The identity $a = b + c$ is stated here."
    llm = "No mathematics appears in this transcription."
    report = detect_divergences("p.pdf", verbatim, llm)
    assert not report.has_blocking  # present-in-one-extractor-only is a coverage artifact, not semantic
    assert report.counts()["notation-only"] >= 1  # still surfaced for review
