"""Structural, type-aware, hierarchical chunking.

Split on headings (parent = section); within a section, group prose to a ~512-token target while
keeping atomic units (theorem/proof/protocol/table/code/math) whole even when oversized. Every leaf
chunk stores a *contiguous verbatim span* so ``verbatim_hash`` audits back to the source; overlap and
section context are applied only to ``embed_input``, never to stored ``content``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import BaseModel

from .config import Settings, get_settings
from .enrich import EnrichmentResult
from .extract.base import ExtractedDoc, span_hash
from .models import Chunk, ChunkType

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_SECTION_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)\.?\s+(.*)$")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[$])")
_PROOF_END_RE = re.compile(r"(∎|□|◻|\bQ\.?E\.?D\.?\b)")

# Default is a neutral academic vocabulary; an instance overrides it with the field's own indivisible
# units (see the "atomic-unit keywords" knob in AGENTS.md). The two buckets are formal statements
# (theorem-like) and named procedures (algorithm/protocol-like).
_ATOMIC_KEYWORDS: dict[str, ChunkType] = {
    "theorem": "theorem", "lemma": "theorem", "proposition": "theorem", "corollary": "theorem",
    "definition": "theorem", "claim": "theorem", "proof": "theorem", "experiment": "theorem",
    "protocol": "protocol", "algorithm": "protocol", "procedure": "protocol",
}
_ATOMIC_RE = re.compile(r"^\s*(?:\*\*)?(" + "|".join(_ATOMIC_KEYWORDS) + r")\b", re.IGNORECASE)


class ParentBlock(BaseModel):
    """A section parent (not embedded) plus its embedded leaf children."""

    parent: Chunk
    children: list[Chunk]


@dataclass
class _Block:
    start: int
    end: int
    text: str
    kind: str  # 'heading' | 'paragraph' | 'code' | 'math' | 'table'
    level: int | None = None
    number: str | None = None
    title: str | None = None


@dataclass
class _Section:
    number: str | None
    title: str
    breadcrumb: str
    heading: _Block | None
    blocks: list[_Block] = field(default_factory=list)


def count_tokens(text: str) -> int:
    """Cheap token estimate (words × 1.3) — good enough for budgeting, not billing."""
    return max(1, round(len(re.findall(r"\S+", text)) * 1.3))


def _atomic_type(text: str) -> ChunkType | None:
    m = _ATOMIC_RE.match(text)
    if not m:
        return None
    return _ATOMIC_KEYWORDS[m.group(1).lower()]


def _iter_blocks(text: str) -> list[_Block]:
    """Tokenize markdown into offset-tracked blocks: headings, code, display math, tables, paragraphs."""
    lines = text.splitlines(keepends=True)
    starts: list[int] = []
    acc = 0
    for ln in lines:
        starts.append(acc)
        acc += len(ln)

    blocks: list[_Block] = []
    i, n = 0, len(lines)
    while i < n:
        raw = lines[i]
        stripped = raw.strip()
        if not stripped:
            i += 1
            continue

        heading = _HEADING_RE.match(raw)
        if heading:
            title = heading.group(2)
            num_m = _SECTION_NUM_RE.match(title)
            number = num_m.group(1) if num_m else None
            clean = num_m.group(2) if (num_m and num_m.group(2)) else title
            blocks.append(
                _Block(starts[i], starts[i] + len(raw), title, "heading",
                       level=len(heading.group(1)), number=number, title=clean)
            )
            i += 1
            continue

        if stripped.startswith("```"):
            j = i + 1
            while j < n and not lines[j].strip().startswith("```"):
                j += 1
            j = min(j, n - 1)
            end = starts[j] + len(lines[j])
            blocks.append(_Block(starts[i], end, text[starts[i]:end], "code"))
            i = j + 1
            continue

        if stripped.startswith("$$") or stripped.startswith("\\["):
            close = "$$" if stripped.startswith("$$") else "\\]"
            j = i
            # single-line $$...$$
            if stripped.count("$$") >= 2 or close in stripped[2:]:
                j = i
            else:
                j = i + 1
                while j < n and close not in lines[j]:
                    j += 1
                j = min(j, n - 1)
            end = starts[j] + len(lines[j])
            blocks.append(_Block(starts[i], end, text[starts[i]:end], "math"))
            i = j + 1
            continue

        if "|" in raw and i + 1 < n and re.match(r"^\s*\|?[\s:|-]*-{2,}[\s:|-]*\|?\s*$", lines[i + 1]):
            j = i
            while j < n and "|" in lines[j] and lines[j].strip():
                j += 1
            end = starts[j - 1] + len(lines[j - 1])
            blocks.append(_Block(starts[i], end, text[starts[i]:end], "table"))
            i = j
            continue

        # paragraph: consecutive non-blank lines until a blank/heading/fence/table starts one
        j = i
        while j < n:
            s = lines[j].strip()
            if not s or _HEADING_RE.match(lines[j]) or s.startswith(("```", "$$", "\\[")):
                break
            j += 1
        end = starts[j - 1] + len(lines[j - 1])
        blocks.append(_Block(starts[i], end, text[starts[i]:end].strip(), "paragraph"))
        i = j
    return blocks


def _split_sections(blocks: list[_Block], doc_title: str) -> list[_Section]:
    """Group blocks into sections (one per heading), tracking a breadcrumb from the heading stack."""
    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []  # (level, title) for breadcrumb
    preamble = _Section(number=None, title="(preamble)", breadcrumb=doc_title, heading=None)
    current = preamble

    for b in blocks:
        if b.kind == "heading":
            level = b.level or 2
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, b.title or b.text))
            breadcrumb = " > ".join([doc_title, *[t for _, t in stack]])
            current = _Section(number=b.number, title=b.title or b.text, breadcrumb=breadcrumb, heading=b)
            sections.append(current)
        else:
            if current is preamble and not sections and not preamble.blocks:
                sections.insert(0, preamble)
            current.blocks.append(b)
    return [s for s in sections if s.blocks or s.heading]


def _split_long_paragraph(block: _Block, target: int, hard_max: int) -> list[tuple[int, int]]:
    """Split an oversized prose block into ≤hard_max spans on sentence boundaries."""
    if count_tokens(block.text) <= hard_max:
        return [(block.start, block.end)]
    spans: list[tuple[int, int]] = []
    cursor = 0
    seg_start = 0
    tokens = 0
    for m in _SENT_SPLIT_RE.finditer(block.text):
        sentence = block.text[cursor:m.start()]
        tokens += count_tokens(sentence)
        cursor = m.start()
        if tokens >= target:
            spans.append((block.start + seg_start, block.start + cursor))
            seg_start = cursor
            tokens = 0
    spans.append((block.start + seg_start, block.end))
    return spans


def _leaf_specs(section: _Section, target: int, hard_max: int) -> list[tuple[int, int, ChunkType]]:
    """Produce (start, end, chunk_type) leaf spans for a section body."""
    specs: list[tuple[int, int, ChunkType]] = []
    buf: list[_Block] = []

    def flush_prose() -> None:
        if not buf:
            return
        specs.append((buf[0].start, buf[-1].end, "paragraph"))
        buf.clear()

    i, blocks = 0, section.blocks
    while i < len(blocks):
        b = blocks[i]
        if b.kind in ("code", "math", "table"):
            flush_prose()
            specs.append((b.start, b.end, b.kind))  # type: ignore[arg-type]
            i += 1
            continue

        atomic = _atomic_type(b.text)
        if atomic:
            flush_prose()
            end = b.end
            if b.text.lower().lstrip("* ").startswith("proof") and not _PROOF_END_RE.search(b.text):
                j = i + 1  # keep a proof whole: absorb following prose until an end marker
                while j < len(blocks) and blocks[j].kind == "paragraph" and not _atomic_type(blocks[j].text):
                    end = blocks[j].end
                    if _PROOF_END_RE.search(blocks[j].text):
                        j += 1
                        break
                    j += 1
                i = j
            else:
                i += 1
            specs.append((b.start, end, atomic))
            continue

        # prose: accumulate to target; split a single oversized paragraph
        if not buf and count_tokens(b.text) > hard_max:
            specs.extend((s, e, "paragraph") for s, e in _split_long_paragraph(b, target, hard_max))
            i += 1
            continue
        buf.append(b)
        if count_tokens("\n\n".join(x.text for x in buf)) >= target:
            flush_prose()
        i += 1
    flush_prose()
    return specs


def _tail(text: str, n_words: int) -> str:
    words = re.findall(r"\S+", text)
    return " ".join(words[-n_words:]) if words else ""


def chunk_document(
    extracted: ExtractedDoc,
    document_id: int,
    title: str,
    enrichment: EnrichmentResult | None = None,
    settings: Settings | None = None,
) -> list[ParentBlock]:
    """Chunk a document into a parent-section / leaf-child tree with provenance and embed inputs."""
    settings = settings or get_settings()
    target, hard_max, overlap = (
        settings.chunk_target_tokens, settings.chunk_hard_max_tokens, settings.chunk_overlap_tokens,
    )
    sections = _split_sections(_iter_blocks(extracted.text), title)

    result: list[ParentBlock] = []
    index = 0
    for section in sections:
        seg_start = section.heading.start if section.heading else (section.blocks[0].start if section.blocks else 0)
        seg_end = section.blocks[-1].end if section.blocks else (section.heading.end if section.heading else 0)
        parent_text = extracted.text[seg_start:seg_end].strip()
        if not parent_text:
            continue
        p_start, p_end = extracted.page_range(seg_start, seg_end)
        parent = Chunk(
            document_id=document_id, chunk_index=index, content=parent_text, content_kind="verbatim",
            section_number=section.number, section_title=section.title, chunk_type="heading",
            page_start=p_start, page_end=p_end, verbatim_hash=span_hash(parent_text),
            token_count=count_tokens(parent_text), embedded=False,
        )
        index += 1

        children: list[Chunk] = []
        prev_content = ""
        summary = enrichment.summary_for(section.title) if enrichment else None
        for start, end, ctype in _leaf_specs(section, target, hard_max):
            content = extracted.text[start:end].strip()
            if not content:
                continue
            c_start, c_end = extracted.page_range(start, end)
            context = f"[{section.breadcrumb}]"
            if summary:
                context += f" {summary}"
            overlap_text = _tail(prev_content, overlap)
            embed_input = f"{context}\n{overlap_text + ' ' if overlap_text else ''}{content}".strip()
            children.append(
                Chunk(
                    document_id=document_id, chunk_index=index, content=content, content_kind="verbatim",
                    section_number=section.number, section_title=section.title, chunk_type=ctype,
                    page_start=c_start, page_end=c_end, verbatim_hash=span_hash(content),
                    token_count=count_tokens(content), embed_input=embed_input, embedded=True,
                )
            )
            prev_content = content
            index += 1

        result.append(ParentBlock(parent=parent, children=children))
    return result


def build_derived_chunk(document_id: int, title: str, enrichment: EnrichmentResult, index: int) -> Chunk | None:
    """One derived-summary chunk per document (concept-level recall for terse sections), clearly marked."""
    if not enrichment.section_summaries:
        return None
    lines = [f"{s.section_number or ''} {s.section_title}: {s.summary}".strip() for s in enrichment.section_summaries]
    content = f"Section summaries for {title}:\n" + "\n".join(lines)
    return Chunk(
        document_id=document_id, chunk_index=index, content=content, content_kind="derived_summary",
        section_title="Section summaries", chunk_type="paragraph", token_count=count_tokens(content),
        embed_input=content, embedded=True, verbatim_hash=None,
    )
