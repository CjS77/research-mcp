"""Corpus discovery over the reference/ root: markdown reference sources and content hashing."""

from __future__ import annotations

from research_kb.config import Settings
from research_kb.corpus import scan, sha256_file


def test_markdown_reference_source(settings: Settings):
    # A plain external text source under the reference/ corpus root (e.g. notes/transcript).
    (settings.reference_dir / "series-analysis-notes.md").write_text(
        "# Notes on Series Analysis\n\nA reference note, not a project research doc.\n",
        encoding="utf-8",
    )
    item = next(i for i in scan(settings) if i.abs_path.name == "series-analysis-notes.md")
    assert item.title == "Notes on Series Analysis"
    assert item.tier == "breadth"
    assert item.phase is None
    assert not item.is_pdf


def test_txt_reference_source(settings: Settings):
    # Canonical RFC .txt files are corpus material: typed as specs, title from the cleaned stem.
    (settings.reference_dir / "rfc7693_blake2_saarinen_aumasson_2015.txt").write_text(
        "Internet Research Task Force (IRTF)\n\nThe BLAKE2 Hash Function.\n",
        encoding="utf-8",
    )
    item = next(i for i in scan(settings) if i.abs_path.name == "rfc7693_blake2_saarinen_aumasson_2015.txt")
    assert item.doc_type == "spec"
    assert item.title == "Rfc7693 Blake2 Saarinen Aumasson 2015"
    assert item.tier == "breadth"
    assert not item.is_pdf


def test_txt_without_rfc_prefix_is_research(settings: Settings):
    (settings.reference_dir / "transcript-notes.txt").write_text("plain notes, no headings\n", encoding="utf-8")
    item = next(i for i in scan(settings) if i.abs_path.name == "transcript-notes.txt")
    assert item.doc_type == "research"


def test_title_from_h1(settings: Settings):
    (settings.reference_dir / "note.md").write_text("# My Considered Title\n\nbody\n")
    item = next(i for i in scan(settings) if i.abs_path.name == "note.md")
    assert item.title == "My Considered Title"


def test_content_hash_changes_with_content(settings: Settings):
    p = settings.reference_dir / "h.md"
    p.write_text("# A\n\none\n")
    first = sha256_file(p)
    p.write_text("# A\n\ntwo\n")
    assert sha256_file(p) != first
