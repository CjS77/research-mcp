"""Named, N-ary facets: profile declaration → enrichment tagging → filtering by any facet name.

Exercises three declared facets (more than the old fixed two) end to end: enrichment tags each
document against the declared vocabularies, the facet values round-trip through the JSON column, and
``kb_search`` filters on any facet by name — with json_each giving exact array membership (no
cross-facet false positives).
"""

from __future__ import annotations

import json

import pytest

from research_kb import mcp_server
from research_kb.config import FacetSpec, Settings
from research_kb.db import connect, get_facet_names, init_db
from research_kb.enrich import heuristic_enrich
from research_kb.index import index_corpus
from research_kb.models import Document
from research_kb.service import list_corpus_service, search_service
from research_kb.store import get_document, upsert_document

from .helpers import make_extracted

# Three declared axes — deliberately N > 2 to prove the shape is no longer hardcoded.
FACETS = (
    FacetSpec(name="clade", terms=("theropod", "sauropod", "ornithischian")),
    FacetSpec(name="period", terms=("triassic", "jurassic", "cretaceous")),
    FacetSpec(name="diet", terms=("carnivore", "herbivore")),
)


@pytest.fixture
def faceted_settings(settings: Settings) -> Settings:
    return settings.model_copy(update={"facets": FACETS})


@pytest.fixture
def faceted_corpus(faceted_settings: Settings) -> Settings:
    """Three dinosaur docs, each exhibiting a distinct mix across the three facets."""
    ref = faceted_settings.reference_dir
    (ref / "tyrannosaurus.md").write_text(
        "# Tyrannosaurus Rex\n\n## 1 Overview\n\n"
        "This dinosaur fossil belongs to a theropod from the cretaceous, an apex carnivore.\n",
        encoding="utf-8",
    )
    (ref / "brachiosaurus.md").write_text(
        "# Brachiosaurus\n\n## 1 Overview\n\n"
        "This dinosaur fossil is a sauropod from the jurassic, a towering herbivore.\n",
        encoding="utf-8",
    )
    (ref / "triceratops.md").write_text(
        "# Triceratops\n\n## 1 Overview\n\n"
        "This dinosaur fossil is an ornithischian from the cretaceous, a horned herbivore.\n",
        encoding="utf-8",
    )
    return faceted_settings


def _papers(hits: list[dict]) -> set[str]:
    return {h["paper"] for h in hits}


def test_heuristic_enrich_tags_all_declared_facets(faceted_settings: Settings):
    extracted = make_extracted("A theropod carnivore from the cretaceous period.")
    result = heuristic_enrich(extracted, "T. Rex", faceted_settings)
    assert result.facets == {"clade": ["theropod"], "period": ["cretaceous"], "diet": ["carnivore"]}


def test_heuristic_enrich_omits_facets_with_no_matches(faceted_settings: Settings):
    # Only the clade vocabulary matches; period/diet are absent → omitted, not stored empty.
    result = heuristic_enrich(make_extracted("A lone sauropod."), "x", faceted_settings)
    assert result.facets == {"clade": ["sauropod"]}


def test_document_facets_roundtrip(faceted_settings: Settings):
    con = init_db(faceted_settings)
    doc = Document(
        source_path="reference/x.md", doc_type="paper", title="X",
        facets={"clade": ["theropod"], "diet": ["carnivore"]},
    )
    doc_id = upsert_document(con, doc)
    con.commit()
    loaded = get_document(con, doc_id)
    assert loaded is not None
    assert loaded.facets == {"clade": ["theropod"], "diet": ["carnivore"]}


def test_indexing_records_declared_facet_names(faceted_corpus: Settings):
    index_corpus(faceted_corpus)
    con = connect(faceted_corpus.db_path)
    assert get_facet_names(con) == ["clade", "period", "diet"]
    # Self-describing: recorded verbatim in kb_meta.
    raw = con.execute("SELECT value FROM kb_meta WHERE key = 'facet_names'").fetchone()[0]
    assert json.loads(raw) == ["clade", "period", "diet"]


def test_indexing_tags_documents(faceted_corpus: Settings):
    index_corpus(faceted_corpus)
    con = connect(faceted_corpus.db_path)
    trex = con.execute("SELECT facets FROM documents WHERE title = 'Tyrannosaurus Rex'").fetchone()[0]
    assert json.loads(trex) == {"clade": ["theropod"], "period": ["cretaceous"], "diet": ["carnivore"]}


def test_filter_by_first_facet(faceted_corpus: Settings):
    index_corpus(faceted_corpus)
    con = connect(faceted_corpus.db_path)
    hits = search_service(con, "dinosaur fossil", filters={"facets": {"clade": ["theropod"]}}, k=10,
                          settings=faceted_corpus)
    assert _papers(hits) == {"Tyrannosaurus Rex"}


def test_filter_by_third_facet_proves_n_ary(faceted_corpus: Settings):
    # 'diet' is a third axis beyond the old fixed facet_a/facet_b pair.
    index_corpus(faceted_corpus)
    con = connect(faceted_corpus.db_path)
    hits = search_service(con, "dinosaur fossil", filters={"facets": {"diet": ["herbivore"]}}, k=10,
                          settings=faceted_corpus)
    assert _papers(hits) == {"Brachiosaurus", "Triceratops"}


def test_filter_shared_value_across_documents(faceted_corpus: Settings):
    index_corpus(faceted_corpus)
    con = connect(faceted_corpus.db_path)
    hits = search_service(con, "dinosaur fossil", filters={"facets": {"period": ["cretaceous"]}}, k=10,
                          settings=faceted_corpus)
    assert _papers(hits) == {"Tyrannosaurus Rex", "Triceratops"}


def test_filter_conjunction_across_facets(faceted_corpus: Settings):
    index_corpus(faceted_corpus)
    con = connect(faceted_corpus.db_path)
    hits = search_service(
        con, "dinosaur fossil",
        filters={"facets": {"clade": ["theropod"], "diet": ["carnivore"]}}, k=10, settings=faceted_corpus,
    )
    assert _papers(hits) == {"Tyrannosaurus Rex"}


def test_facet_path_is_exact_no_cross_facet_bleed(faceted_corpus: Settings):
    # 'cretaceous' is a *period* value; filtering it under 'clade' must match nothing.
    index_corpus(faceted_corpus)
    con = connect(faceted_corpus.db_path)
    hits = search_service(con, "dinosaur fossil", filters={"facets": {"clade": ["cretaceous"]}}, k=10,
                          settings=faceted_corpus)
    assert hits == []


def test_list_corpus_advertises_facets(faceted_corpus: Settings):
    index_corpus(faceted_corpus)
    con = connect(faceted_corpus.db_path)
    result = list_corpus_service(con)
    assert result["facets"] == ["clade", "period", "diet"]
    trex = next(d for d in result["documents"] if d["title"] == "Tyrannosaurus Rex")
    assert trex["facets"] == {"clade": ["theropod"], "period": ["cretaceous"], "diet": ["carnivore"]}


def test_mcp_kb_search_facets_mapping(faceted_corpus: Settings, monkeypatch: pytest.MonkeyPatch):
    index_corpus(faceted_corpus)
    monkeypatch.setattr(mcp_server, "get_settings", lambda: faceted_corpus)
    hits = mcp_server.kb_search("dinosaur fossil", k=10, facets={"clade": "theropod"})
    assert {h["paper"] for h in hits} == {"Tyrannosaurus Rex"}
