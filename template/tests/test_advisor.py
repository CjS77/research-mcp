"""The embedding-model advisor: representative corpus signals map to the expected model.

All heuristics are offline (no model download, no network) so these run fast in CI.
"""

from __future__ import annotations

from research_kb.advisor import (
    DENSE_EN,
    GENERAL_EN,
    MULTILINGUAL,
    advise,
    analyze_text,
    dim_for_model,
    recommend,
)
from research_kb.config import Settings
from research_kb.db import connect
from research_kb.index import index_corpus

# --- Representative corpus samples ---------------------------------------------------------------

GENERAL_ENGLISH = (
    "The museum opened a new hall about dinosaurs last spring. Visitors can walk among the big "
    "skeletons and read short notes about how these animals lived. Children love the moving models, "
    "and the gift shop sells toys shaped like a stegosaurus. It is a nice place to spend an afternoon "
    "with the family, and the cafe by the door serves good coffee and cake to everyone who comes in."
)

DENSE_ACADEMIC = (
    "Abstract. We prove a convergence theorem for the proposed estimator under mild regularity "
    "assumptions [1]. Lemma 2 establishes that the asymptotic variance of the coefficient is bounded, "
    "and the corollary follows by a standard argument (Smith et al., 2019). Our empirical methodology "
    "evaluates the algorithm on a benchmark dataset; the regression analysis reports the parameters in "
    "the appendix. The equation $\\sum_{i=1}^{n} x_i \\leq \\alpha$ characterises the proof, and "
    "figure 3 shows the significant asymptotic behaviour of the estimator respectively [2] [3]."
)

NON_ENGLISH_LATIN = (
    "Le musee a ouvert une nouvelle salle sur les dinosaures au printemps dernier. Les visiteurs "
    "peuvent marcher parmi les grands squelettes et lire de courtes notes sur la maniere dont ces "
    "animaux vivaient. Les enfants adorent les modeles qui bougent, et la boutique vend des jouets. "
    "C'est un bel endroit pour passer un apres-midi en famille, avec un cafe pres de la porte."
)

NON_ENGLISH_CYRILLIC = (
    "Музей открыл новый зал о динозаврах прошлой весной. Посетители могут ходить среди больших "
    "скелетов и читать короткие заметки о том, как жили эти животные. Дети любят движущиеся модели, "
    "а магазин продаёт игрушки. Это хорошее место, чтобы провести день с семьёй у входа в кафе."
)


def test_general_english_recommends_default():
    rec = recommend(analyze_text(GENERAL_ENGLISH))
    assert rec.model == GENERAL_EN.model
    assert rec.dim == GENERAL_EN.dim
    assert rec.signals.non_latin_ratio < 0.05
    assert rec.signals.english_stopword_ratio > 0.15
    assert rec.signals.academic_score < 0.35


def test_dense_academic_recommends_larger_model():
    rec = recommend(analyze_text(DENSE_ACADEMIC))
    assert rec.model == DENSE_EN.model
    assert rec.signals.academic_score >= 0.35
    assert rec.signals.english_stopword_ratio >= 0.08  # still English, just dense


def test_non_english_latin_recommends_multilingual():
    # French: Latin script (low non-Latin ratio) but poor in English stopwords -> multilingual.
    rec = recommend(analyze_text(NON_ENGLISH_LATIN))
    assert rec.model == MULTILINGUAL.model
    assert rec.signals.non_latin_ratio < 0.05
    assert rec.signals.english_stopword_ratio < 0.08


def test_non_english_cyrillic_recommends_multilingual():
    rec = recommend(analyze_text(NON_ENGLISH_CYRILLIC))
    assert rec.model == MULTILINGUAL.model
    assert rec.signals.non_latin_ratio > 0.10


def test_dim_for_model_known_and_unknown():
    assert dim_for_model(GENERAL_EN.model) == GENERAL_EN.dim
    assert dim_for_model(MULTILINGUAL.model) == MULTILINGUAL.dim
    assert dim_for_model("some/unknown-model") is None


def test_advise_reads_reference_files(settings: Settings):
    (settings.reference_dir / "french-note.md").write_text(NON_ENGLISH_LATIN, encoding="utf-8")
    rec = advise(settings, con=None, prefer="reference")
    assert rec.model == MULTILINGUAL.model
    assert rec.signals.source == "reference"
    assert rec.signals.n_docs == 1


def test_advise_reads_indexed_chunks(small_corpus: Settings):
    # small_corpus is dense English (theorems, citations) -> the larger model; source is the DB.
    index_corpus(small_corpus)
    con = connect(small_corpus.db_path)
    rec = advise(small_corpus, con, prefer="db")
    assert rec.model == DENSE_EN.model
    assert rec.signals.source == "db"
    assert rec.signals.n_docs >= 1


def test_as_dict_rounds_signals():
    d = recommend(analyze_text(GENERAL_ENGLISH)).as_dict()
    assert set(d) == {"model", "dim", "rationale", "signals"}
    sig = d["signals"]
    assert isinstance(sig, dict)
    assert isinstance(sig["academic_score"], float)
