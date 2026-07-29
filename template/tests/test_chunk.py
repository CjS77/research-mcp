"""Chunker: section split, atomic-unit rule, type-awareness, provenance, embed context."""

from __future__ import annotations

from research_kb.chunk import chunk_document, count_tokens
from research_kb.extract.base import span_hash
from tests.helpers import make_extracted

DOC = """# Paper Title

## 1 Introduction

This is the first paragraph of the introduction with enough words to be a real chunk of prose.

## 2 Results

**Theorem 1.** The estimator is consistent across the sample and this atomic statement is long enough
that it must be preserved whole because splitting a theorem across chunks would break its meaning.

```python
def scale(x):
    return x * 2
```

$$ E = m c^2 $$
"""


def test_sections_and_children_built():
    extracted = make_extracted(DOC)
    tree = chunk_document(extracted, document_id=1, title="Paper Title")
    titles = [pb.parent.section_title for pb in tree]
    assert "Introduction" in titles and "Results" in titles


def test_atomic_types_preserved_and_not_split():
    extracted = make_extracted(DOC)
    tree = chunk_document(extracted, document_id=1, title="Paper Title")
    children = [c for pb in tree for c in pb.children]
    kinds = {c.chunk_type for c in children}
    assert {"theorem", "code", "math"} <= kinds
    theorem = next(c for c in children if c.chunk_type == "theorem")
    assert "must be preserved whole" in theorem.content  # kept intact


def test_verbatim_hash_matches_content():
    extracted = make_extracted(DOC)
    tree = chunk_document(extracted, document_id=1, title="Paper Title")
    for pb in tree:
        for c in [pb.parent, *pb.children]:
            assert c.verbatim_hash == span_hash(c.content)


def test_child_pages_and_embed_context():
    extracted = make_extracted(DOC, page_len=120)
    tree = chunk_document(extracted, document_id=1, title="Paper Title")
    child = next(c for pb in tree for c in pb.children if c.chunk_type == "paragraph")
    assert child.page_start is not None and child.page_end is not None
    # embed_input carries the section breadcrumb but stored content stays verbatim
    assert child.embed_input.startswith("[Paper Title")
    assert child.content in DOC


def test_code_block_not_paraphrased():
    extracted = make_extracted(DOC)
    tree = chunk_document(extracted, document_id=1, title="Paper Title")
    code = next(c for pb in tree for c in pb.children if c.chunk_type == "code")
    assert "def scale(x):" in code.content


def test_count_tokens_positive():
    assert count_tokens("") == 1
    assert count_tokens("a b c d") >= 4
