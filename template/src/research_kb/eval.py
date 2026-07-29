"""Evaluation harness: the arbiter for every tuning decision.

Metrics: ``recall@k`` and ``MRR`` over a gold-query set (document-level target unless a section/page
is specified), plus a ``faithfulness`` check asserting every stored ``verbatim`` chunk still matches
its recorded ``verbatim_hash``. Run it on every model / chunk-size / weighting change — without it,
recall targets are unmeasurable.
"""

from __future__ import annotations

import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import Settings, get_settings
from .db import connect
from .embed import get_query_embedder
from .embed.base import EmbeddingProvider
from .extract.base import span_hash
from .models import EvalQuery
from .search import hybrid_search
from .store import clear_eval_queries, get_document_by_path, insert_eval_query, list_eval_queries


@dataclass
class QueryResult:
    query: str
    expected_document_id: int | None
    hit_rank: int | None  # 1-based rank of first chunk from the expected document, else None
    retrieved_papers: list[str] = field(default_factory=list)

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.hit_rank if self.hit_rank else 0.0


@dataclass
class EvalReport:
    k: int
    n_queries: int
    recall_at_k: float
    mrr: float
    faithfulness_checked: int
    faithfulness_matched: int
    per_query: list[QueryResult] = field(default_factory=list)

    @property
    def faithfulness_ratio(self) -> float:
        return self.faithfulness_matched / self.faithfulness_checked if self.faithfulness_checked else 1.0

    def as_dict(self) -> dict[str, object]:
        return {
            "k": self.k, "n_queries": self.n_queries, "recall_at_k": round(self.recall_at_k, 4),
            "mrr": round(self.mrr, 4), "faithfulness_ratio": round(self.faithfulness_ratio, 6),
            "faithfulness_checked": self.faithfulness_checked,
        }


def load_gold_queries(con: sqlite3.Connection, path: Path, settings: Settings | None = None) -> int:
    """Load a gold-query YAML file into ``eval_queries``, resolving expected documents by path."""
    settings = settings or get_settings()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    clear_eval_queries(con)
    loaded = 0
    for entry in data:
        expected = get_document_by_path(con, entry["expected_source_path"]) if entry.get("expected_source_path") else None
        insert_eval_query(
            con,
            EvalQuery(
                query=entry["query"],
                expected_document_id=expected.id if expected else None,
                expected_section=str(entry["expected_section"]) if entry.get("expected_section") else None,
                expected_page=entry.get("expected_page"),
                notes=entry.get("notes"),
            ),
        )
        loaded += 1
    con.commit()
    return loaded


def faithfulness_check(con: sqlite3.Connection) -> tuple[int, int]:
    """(checked, matched) over verbatim chunks: content must still hash to its recorded verbatim_hash."""
    rows = con.execute(
        "SELECT content, verbatim_hash FROM chunks WHERE content_kind = 'verbatim' AND verbatim_hash IS NOT NULL"
    ).fetchall()
    matched = sum(1 for r in rows if span_hash(r["content"]) == r["verbatim_hash"])
    return len(rows), matched


def run_eval(
    con: sqlite3.Connection,
    embedder: EmbeddingProvider | None = None,
    k: int = 10,
    settings: Settings | None = None,
) -> EvalReport:
    """Run the gold set and compute recall@k, MRR, and the faithfulness check."""
    settings = settings or get_settings()
    embedder = embedder or get_query_embedder(con, settings)
    queries = [q for q in list_eval_queries(con) if q.expected_document_id is not None]

    per_query: list[QueryResult] = []
    hits = 0
    rr_sum = 0.0
    for q in queries:
        results = hybrid_search(con, q.query, embedder=embedder, k=k, settings=settings)
        rank = next((i for i, h in enumerate(results, start=1) if h.document_id == q.expected_document_id), None)
        if rank is not None:
            hits += 1
            rr_sum += 1.0 / rank
        per_query.append(
            QueryResult(
                query=q.query, expected_document_id=q.expected_document_id, hit_rank=rank,
                retrieved_papers=list(dict.fromkeys(h.paper for h in results)),
            )
        )

    n = len(queries)
    checked, matched = faithfulness_check(con)
    return EvalReport(
        k=k, n_queries=n,
        recall_at_k=hits / n if n else 0.0,
        mrr=rr_sum / n if n else 0.0,
        faithfulness_checked=checked, faithfulness_matched=matched,
        per_query=per_query,
    )


# --- A/B: compare two embedding models on the same gold set --------------------------------------
# Switching the embedding model is a rebuild, not a migration (the DB self-describes its vector
# space via kb_meta). So A/B indexes each model into its *own* fresh DB, then runs the same gold
# queries against each — the primary index is never mutated.


@dataclass
class ABModelSpec:
    """One arm of an A/B run: the embedding backend/model/dimension to build and score."""

    model: str
    dim: int
    backend: str = "fastembed"
    label: str = ""

    def display(self) -> str:
        return self.label or self.model


@dataclass
class ABReport:
    """Two eval reports over one gold set, one per embedding model."""

    k: int
    spec_a: ABModelSpec
    report_a: EvalReport
    spec_b: ABModelSpec
    report_b: EvalReport

    def as_dict(self) -> dict[str, object]:
        return {
            "k": self.k,
            "a": {"model": self.spec_a.model, "dim": self.spec_a.dim, "backend": self.spec_a.backend,
                  **self.report_a.as_dict()},
            "b": {"model": self.spec_b.model, "dim": self.spec_b.dim, "backend": self.spec_b.backend,
                  **self.report_b.as_dict()},
        }


def _safe_label(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in text).strip("-") or "model"


def _eval_one_model(
    settings: Settings,
    gold_path: Path,
    spec: ABModelSpec,
    k: int,
    workdir: Path,
) -> EvalReport:
    """Rebuild the corpus under one embedding model into an isolated DB, then run the gold set."""
    from .index import index_corpus  # local import: index -> embed pulls heavy deps only when A/B runs

    arm_dir = workdir / _safe_label(spec.display())
    arm_dir.mkdir(parents=True, exist_ok=True)
    arm_settings = settings.model_copy(
        update={
            "embed_backend": spec.backend,
            "embed_model": spec.model,
            "embed_dim": spec.dim,
            "db_path": arm_dir / "kb.sqlite",
            "distilled_dir": arm_dir / "distilled",
        }
    )
    index_corpus(arm_settings)
    con = connect(arm_settings.db_path)
    try:
        load_gold_queries(con, gold_path, arm_settings)
        return run_eval(con, k=k, settings=arm_settings)
    finally:
        con.close()


def run_ab_eval(
    settings: Settings,
    gold_path: Path,
    spec_a: ABModelSpec,
    spec_b: ABModelSpec,
    k: int = 10,
    workdir: Path | None = None,
) -> ABReport:
    """Build and score two embedding models on the same gold queries, side by side.

    Each model is indexed into its own throwaway DB (a rebuild), so neither the primary index nor the
    other arm is touched. Model choice stays eval-gated: this returns the numbers; the caller decides.
    """
    if workdir is not None:
        workdir.mkdir(parents=True, exist_ok=True)
        report_a = _eval_one_model(settings, gold_path, spec_a, k, workdir)
        report_b = _eval_one_model(settings, gold_path, spec_b, k, workdir)
    else:
        with tempfile.TemporaryDirectory(prefix="kb-ab-eval-") as tmp:
            root = Path(tmp)
            report_a = _eval_one_model(settings, gold_path, spec_a, k, root)
            report_b = _eval_one_model(settings, gold_path, spec_b, k, root)
    return ABReport(k=k, spec_a=spec_a, report_a=report_a, spec_b=spec_b, report_b=report_b)
