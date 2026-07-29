"""Citation extraction, resolution, and acquisition targets."""

from __future__ import annotations

from research_kb.citations import extract_citations
from tests.helpers import make_extracted

REFS = """# Some Paper

## 1 Body

Text that cites prior work.

## References

[1] A. Author. The Beta Method for Signal Denoising. Journal, 2020.
[2] B. Writer. Some Unrelated Database Construction. Conf, 2019.
[3] C. Person. Another Estimation Result. 2021.
"""


def test_extract_citations_splits_bracketed_entries():
    extracted = make_extracted(REFS, page_len=10_000)
    cites = extract_citations(extracted)
    assert len(cites) == 3
    assert any("Beta Method for Signal Denoising" in ref for ref, _page in cites)


def test_no_references_section_yields_nothing():
    extracted = make_extracted("# Paper\n\n## 1 Intro\n\nNo bibliography here.\n")
    assert extract_citations(extracted) == []
