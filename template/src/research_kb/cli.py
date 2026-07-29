"""``research-kb`` command-line interface (the secondary surface).

A thin wrapper over the same core library the MCP server uses: index / search / status / eval /
paper / citations / validate. Human-readable by default; ``--json`` where structured output helps.
"""

from __future__ import annotations

import json as _json
from pathlib import Path

import click

from .config import get_settings
from .db import connect, get_meta, init_db
from .eval import load_gold_queries, run_eval
from .index import index_corpus
from .service import (
    follow_citations_service,
    get_paper_service,
    list_corpus_service,
    search_service,
)
from .store import get_document_by_path, set_validated


def _con():
    return connect(get_settings().db_path)


@click.group()
def main() -> None:
    """Fidelity-first hybrid-retrieval knowledge base over a corpus of documents."""


@main.command()
def init() -> None:
    """Create the SQLite KB and schema."""
    s = get_settings()
    init_db(s)
    click.echo(f"initialized {s.db_path} (embed backend={s.embed_backend}, dim={s.embed_dim})")


@main.command()
@click.option("--scan", "scans", multiple=True, type=click.Path(), help="Root(s) to scan; repeatable.")
@click.option("--force", is_flag=True, help="Re-index even if content_hash is unchanged.")
@click.option("--no-embed", is_flag=True, help="Chunk + index without computing embeddings.")
def index(scans: tuple[str, ...], force: bool, no_embed: bool) -> None:
    """Distil + chunk + embed + index the corpus."""
    s = get_settings()
    roots = [Path(p) for p in scans] or None
    summary = index_corpus(s, roots=roots, force=force, embed=not no_embed)
    click.echo(
        f"indexed={len(summary.indexed)} skipped={len(summary.skipped)} "
        f"blocked={len(summary.blocked)} pruned={len(summary.pruned)} failed={len(summary.failed)}"
    )
    click.echo(f"chunks={summary.chunks_created} citations_resolved={summary.citations_resolved}")
    for path, err in summary.failed:
        click.echo(f"  FAILED {path}: {err}", err=True)
    for path in summary.blocked:
        click.echo(f"  BLOCKED (semantic divergence) {path} — run `research-kb validate {path}`", err=True)


@main.command()
@click.argument("query")
@click.option("-k", "k", default=8, help="Number of results.")
@click.option("--tier", type=click.Choice(["core", "breadth"]))
@click.option("--doc-type", "doc_type")
@click.option("--phase", type=int)
@click.option("--facet-a", "facet_a", multiple=True)
@click.option("--facet-b", "facet_b", multiple=True)
@click.option("--json", "as_json", is_flag=True)
def search(query, k, tier, doc_type, phase, facet_a, facet_b, as_json) -> None:
    """Hybrid semantic + keyword search."""
    filters: dict[str, object] = {}
    if tier:
        filters["tier"] = tier
    if doc_type:
        filters["doc_type"] = doc_type
    if phase is not None:
        filters["phase"] = phase
    if facet_a:
        filters["facet_a"] = list(facet_a)
    if facet_b:
        filters["facet_b"] = list(facet_b)

    con = _con()
    hits = search_service(con, query, filters=filters or None, k=k)
    if as_json:
        click.echo(_json.dumps(hits, indent=2))
        return
    if not hits:
        click.echo("no results")
        return
    for h in hits:
        loc = f"p{h['page_start']}" if h["page_start"] else (h["section_number"] or "")
        sect = h["section_number"] or "-"
        click.echo(
            f"[{h['score']:.4f}] {h['paper'][:44]}  §{sect} {loc}  "
            f"({h['content_kind']}/{h['chunk_type']}, chunk {h['chunk_id']})"
        )
        click.echo(f"    {h['snippet'][:200]}")


@main.command()
@click.option("--json", "as_json", is_flag=True)
def status(as_json: bool) -> None:
    """Corpus + index summary."""
    con = _con()
    docs = con.execute("SELECT tier, doc_type, COUNT(*) n FROM documents GROUP BY tier, doc_type").fetchall()
    chunk_kinds = con.execute("SELECT content_kind, COUNT(*) n FROM chunks GROUP BY content_kind").fetchall()
    totals = con.execute(
        "SELECT (SELECT COUNT(*) FROM documents) d, (SELECT COUNT(*) FROM chunks) c, "
        "(SELECT COUNT(*) FROM chunks WHERE embedded=1) e, (SELECT COUNT(*) FROM documents WHERE validated=1) v"
    ).fetchone()
    cites = con.execute(
        "SELECT COUNT(*) t, SUM(CASE WHEN to_document_id IS NOT NULL THEN 1 ELSE 0 END) r FROM citations"
    ).fetchone()
    jobs = con.execute("SELECT status, COUNT(*) n FROM indexing_jobs GROUP BY status").fetchall()

    if as_json:
        click.echo(_json.dumps({
            "documents": totals["d"], "chunks": totals["c"], "embedded": totals["e"], "validated": totals["v"],
            "citations": cites["t"], "citations_resolved": cites["r"] or 0,
            "embed_backend": get_meta(con, "embed_backend"), "embed_dim": get_meta(con, "embed_dim"),
        }, indent=2))
        return

    click.echo(f"documents: {totals['d']}  chunks: {totals['c']}  embedded: {totals['e']}  validated: {totals['v']}")
    click.echo(f"embed backend: {get_meta(con, 'embed_backend')} (dim {get_meta(con, 'embed_dim')})")
    click.echo(f"citations: {cites['t']} ({cites['r'] or 0} resolved)")
    click.echo("by tier/type:")
    for r in docs:
        click.echo(f"  {r['tier']:8} {r['doc_type']:11} {r['n']}")
    click.echo("chunk kinds: " + ", ".join(f"{r['content_kind']}={r['n']}" for r in chunk_kinds))
    click.echo("jobs: " + ", ".join(f"{r['status']}={r['n']}" for r in jobs))


@main.command()
@click.option("--gold", type=click.Path(exists=True), default="work/eval/gold_queries.yaml")
@click.option("-k", "k", default=10)
@click.option("--json", "as_json", is_flag=True)
def eval(gold: str, k: int, as_json: bool) -> None:
    """Load the gold set and report recall@k / MRR / faithfulness."""
    s = get_settings()
    con = _con()
    n = load_gold_queries(con, Path(gold), s)
    report = run_eval(con, k=k, settings=s)
    if as_json:
        click.echo(_json.dumps(report.as_dict(), indent=2))
        return
    click.echo(f"gold queries: {n} (evaluable: {report.n_queries})")
    click.echo(f"recall@{k}: {report.recall_at_k:.3f}   MRR: {report.mrr:.3f}")
    click.echo(f"faithfulness: {report.faithfulness_matched}/{report.faithfulness_checked} "
               f"({report.faithfulness_ratio:.4f})")
    misses = [q for q in report.per_query if q.hit_rank is None]
    if misses:
        click.echo(f"misses ({len(misses)}):")
        for q in misses:
            click.echo(f"  - {q.query[:72]}")


@main.command()
@click.argument("identifier")
@click.option("--json", "as_json", is_flag=True)
def paper(identifier: str, as_json: bool) -> None:
    """Show a document's metadata + section outline + artifact links."""
    con = _con()
    info = get_paper_service(con, identifier)
    if info is None:
        raise click.ClickException(f"no paper matching {identifier!r}")
    if as_json:
        click.echo(_json.dumps(info, indent=2))
        return
    click.echo(f"[{info['id']}] {info['title']}")
    click.echo(f"  {info['doc_type']}/{info['tier']}  year={info['year']}  pages={info['page_count']}  "
               f"validated={info['validated']}")
    click.echo(f"  facet_a: {', '.join(info['facet_a']) or '-'}")
    click.echo(f"  artifacts: {', '.join(info['artifacts'].keys()) or '-'}")
    click.echo(f"  sections ({len(info['section_outline'])}):")
    for s in info["section_outline"][:40]:
        pg = f" p{s['page_start']}" if s["page_start"] else ""
        click.echo(f"    {s['section_number'] or '-':7} {s['section_title']}{pg}")


@main.command()
@click.argument("document_id", type=int)
@click.option("--direction", type=click.Choice(["out", "in"]), default="out")
def citations(document_id: int, direction: str) -> None:
    """Traverse citation edges out of / into a document."""
    con = _con()
    result = follow_citations_service(con, document_id, direction)
    if result is None:
        raise click.ClickException(f"no document with id {document_id}")
    click.echo(f"{result['document']['title']} — citations {direction}:")
    for e in result["edges"]:
        if direction == "out":
            tag = f"→ [{e['resolved']['id']}] {e['resolved']['title'][:44]}" if e["resolved"] else "→ (not in corpus)"
            click.echo(f"  {tag}\n     {e['to_reference'][:90]}")
        else:
            click.echo(f"  ← [{e['from']['id']}] {e['from']['title'][:50]}")


@main.command()
@click.option("--tier", type=click.Choice(["core", "breadth"]))
@click.option("--acquire", is_flag=True, help="Include cited-but-missing acquisition targets.")
@click.option("--json", "as_json", is_flag=True)
def corpus(tier: str, acquire: bool, as_json: bool) -> None:
    """List indexed documents; optionally the acquisition targets."""
    con = _con()
    result = list_corpus_service(con, filters={"tier": tier} if tier else None, include_acquisition=acquire)
    if as_json:
        click.echo(_json.dumps(result, indent=2))
        return
    click.echo(f"{result['count']} documents:")
    for d in result["documents"]:
        click.echo(f"  [{d['id']:2}] {d['tier']:7} {d['doc_type']:11} {d['chunks']:4}ch  {d['title'][:52]}")
    if acquire:
        click.echo("\ncited-but-missing (acquisition targets):")
        for t in result.get("acquisition_targets", []):
            click.echo(f"  x{t['citations']}  {t['reference'][:80]}")


@main.command()
@click.argument("source_path")
@click.option("--unset", is_flag=True, help="Clear the validated flag instead of setting it.")
def validate(source_path: str, unset: bool) -> None:
    """Mark a (core-tier) document human-validated so it clears the divergence gate."""
    con = _con()
    doc = get_document_by_path(con, source_path)
    if doc is None or doc.id is None:
        raise click.ClickException(f"no document at {source_path!r}")
    set_validated(con, doc.id, not unset)
    con.commit()
    click.echo(f"{'validated' if not unset else 'unvalidated'}: {doc.title}")


@main.command()
@click.option("--manifest", type=click.Path(exists=True), required=True, help="YAML: [{filename, title, url}].")
@click.option("--dest", type=click.Path(), default="reference", help="Where verified PDFs land.")
def acquire(manifest: str, dest: str) -> None:
    """Download + content-verify papers from a manifest. Wrong papers are rejected."""
    from .acquire import acquire_from_manifest

    def live(status: str, filename: str, detail: str) -> None:
        if status == "OK":
            click.echo(f"  OK   {filename} ({detail})")
        elif status == "SKIP":
            click.echo(f"  SKIP {filename}")
        else:
            click.echo(f"  REJ  {filename}: {detail}", err=True)

    result = acquire_from_manifest(Path(manifest), Path(dest), on_result=live)
    click.echo(f"acquired={len(result.acquired)} skipped={len(result.skipped)} rejected={len(result.rejected)}")


@main.command()
@click.argument("query")
@click.option("--provider", "-p", "providers", multiple=True,
              help="Provider(s) to query; repeatable. Default: all registered.")
@click.option("--since", type=str, default=None, help="Lower bound on submission date, YYYY-MM-DD.")
@click.option("--limit", type=int, default=50, help="Max candidates per provider.")
@click.option("--manifest", type=click.Path(), default="work/acquire-manifest.yaml",
              help="Manifest to merge candidates into (acquire reads this).")
@click.option("--refresh", "do_refresh", is_flag=True,
              help="Incremental: use each provider's persisted last-run as --since and advance it.")
@click.option("--dry-run", is_flag=True, help="Print candidates without writing the manifest.")
@click.option("--json", "as_json", is_flag=True)
def discover(query, providers, since, limit, manifest, do_refresh, dry_run, as_json) -> None:
    """Find candidate documents from provider APIs and merge them into an acquire manifest.

    Queries each provider (arXiv/Crossref/Semantic Scholar) and writes {filename, title, url} entries
    that `acquire` then downloads and content-verifies — discovery never bypasses verification. With
    --refresh, only material newer than each provider's last run is fetched (the cron path).
    """
    from datetime import date

    from . import discovery

    names = list(providers) or discovery.provider_names()
    if do_refresh and since:
        raise click.ClickException("--since and --refresh are mutually exclusive (refresh derives its own since)")

    if do_refresh:
        candidates = discovery.refresh(query, names, get_settings(), limit=limit)
    else:
        since_date = date.fromisoformat(since) if since else None
        candidates = discovery.discover(query, names, since=since_date, limit=limit)

    if as_json:
        click.echo(_json.dumps([c.manifest_entry() for c in candidates], indent=2))
    else:
        counts: dict[str, int] = {}
        for c in candidates:
            counts[c.provider] = counts.get(c.provider, 0) + 1
            click.echo(f"  {c.provider:16} {c.title[:70]}")
        click.echo("found " + ", ".join(f"{n}={counts.get(n, 0)}" for n in names) + f" (total {len(candidates)})")

    if dry_run:
        return
    added, total = discovery.write_manifest(candidates, Path(manifest))
    click.echo(f"manifest {manifest}: +{added} new (total {total})")


def _resolve_pdf(target: str, settings) -> Path | None:
    """Map a distill target (a PDF path or a bare stem) to an existing PDF under reference/."""
    p = Path(target)
    if p.suffix.lower() == ".pdf" and p.exists():
        return p
    candidate = settings.reference_dir / (p.name if p.suffix else f"{target}.pdf")
    return candidate if candidate.exists() else None


@main.command()
@click.argument("targets", nargs=-1)
@click.option("--all", "all_pdfs", is_flag=True, help="Distill every PDF under reference/ lacking an llm.md.")
@click.option("--force", is_flag=True, help="Regenerate llm.md even if it already exists.")
@click.option("--segment-pages", type=int, default=None, help="Pages per transcription segment.")
@click.option("--model", default=None, help="Model alias for the distill backend (claude_cli default: sonnet).")
@click.option("--effort", default=None, help="Reasoning effort where supported: low|medium|high|xhigh|max.")
def distill(targets, all_pdfs, force, segment_pages, model, effort) -> None:
    """Transcribe PDFs to distilled/<stem>/llm.md via the configured distill backend.

    The default backend is headless `claude -p` (subscription auth); select another with
    KB_LLM_EXTRACT_BACKEND. Produces the independent LLM extraction that `index` cross-checks against
    the deterministic transcription. TARGETS are PDF paths or bare stems; use --all for the whole
    reference/ corpus. Skips papers that already have an llm.md unless --force.
    """
    from .extract.backends import get_backend
    from .extract.llm import generate_llm_artifact

    # --model/--effort tune the claude_cli backend's config; future backends add their own knobs.
    overrides = {k: v for k, v in {
        "llm_segment_pages": segment_pages, "claude_model": model, "claude_effort": effort,
    }.items() if v is not None}
    s = get_settings().model_copy(update=overrides)

    backend = get_backend(s.llm_extract_backend)
    if backend is None:
        raise click.ClickException(f"unknown distill backend '{s.llm_extract_backend}' (set KB_LLM_EXTRACT_BACKEND)")
    if not backend.available(s):
        raise click.ClickException(f"distill backend '{backend.name}' unavailable: {backend.unavailable_hint(s)}")

    if all_pdfs:
        pdfs = sorted(s.reference_dir.glob("*.pdf"))
    else:
        pdfs = []
        for t in targets:
            resolved = _resolve_pdf(t, s)
            if resolved is None:
                click.echo(f"  SKIP {t}: no matching PDF", err=True)
            else:
                pdfs.append(resolved)
    if not pdfs:
        raise click.ClickException("no PDFs to distill (give TARGETS or --all)")

    done = skipped = failed = 0
    for pdf in pdfs:
        stem = pdf.stem
        artifact = s.artifact_dir(stem) / "llm.md"
        if artifact.exists() and not force:
            click.echo(f"  have {stem} (llm.md exists; --force to regenerate)")
            skipped += 1
            continue
        click.echo(
            f"  distilling {stem} (backend={s.llm_extract_backend}, {s.claude_model}/{s.claude_effort}, "
            f"{s.llm_segment_pages}p/segment)…"
        )
        result = generate_llm_artifact(pdf, stem, s, force=force)
        if result is None:
            click.echo(f"  FAILED {stem}: extraction returned nothing", err=True)
            failed += 1
        else:
            click.echo(f"  OK   {stem} -> {result}")
            done += 1
    click.echo(f"distilled={done} skipped={skipped} failed={failed}")


@main.command()
def serve() -> None:
    """Start the MCP server (stdio). Equivalent to `research-kb-mcp`."""
    from .mcp_server import main as mcp_main

    mcp_main()


if __name__ == "__main__":
    main()
