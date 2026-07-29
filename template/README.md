# research-kb

A fidelity-first, hybrid-retrieval knowledge-base engine, served over MCP. Point it at a directory
of PDFs and text files; it builds a provenance-carrying index you can search, expand, and traverse —
from Claude Code or any MCP client, or from the CLI.

"Fidelity-first" means the pipeline never paraphrases the source. Each PDF is transcribed twice — by
a deterministic extractor and, optionally, by an LLM — and the two are cross-checked. Only verbatim
spans are stored for quoting; every derived summary is marked as such and every chunk hashes back to
its source span.

## What it does

- **Extract** — deterministic transcription (the source of truth) plus an optional independent LLM
  transcription, with a divergence cross-check that blocks a core source on an unresolved semantic
  mismatch until a human signs off.
- **Chunk** — structural, type-aware, hierarchical chunking that keeps atomic units (theorems,
  tables, code, math) whole and carries section + page provenance on every chunk.
- **Search** — hybrid retrieval: semantic (sqlite-vec) fused with keyword (FTS5 BM25) via reciprocal
  rank fusion. The whole store is a single SQLite file.
- **Traverse** — a citation graph over the corpus, plus a ranked list of cited-but-missing works to
  acquire next.
- **Evaluate** — a gold-query harness reporting recall@k, MRR, and a faithfulness check.

## Quick start

```bash
uv sync
# Drop your PDFs / .md / .txt files under reference/, then:
uv run research-kb init            # create the SQLite KB + schema
uv run research-kb index           # extract -> chunk -> embed -> index the corpus
uv run research-kb search "your query here" -k 8
uv run research-kb status          # corpus + index summary
```

Optionally distil PDFs to a committed LLM transcription first (improves math/layout fidelity):

```bash
uv run research-kb distill --all   # writes work/distilled/<stem>/llm.md
```

## Drafting the domain profile

The engine is corpus-agnostic; a handful of knobs encode domain knowledge (facets, atomic-unit
keywords, claim markers, the extraction prompt, the notation note — see "What must be domain-tuned"
in the top-level playbook). `profile-init` proposes values for all of them from a one-line topic
description, so you edit rather than author from scratch:

```bash
uv run research-kb profile-init "your topic description"   # → work/profile-draft.md (a proposal)
```

The draft is **never applied** — it lands in `work/profile-draft.md` as paste-ready Python snippets
keyed to each knob's home file, for you to review, tune, and copy in. It uses the configured LLM
backend (`KB_LLM_EXTRACT_BACKEND`) when one is available and degrades to a fill-in scaffold offline.

## Discovering sources

Turn a topic query into a **verified-acquisition manifest** without hand-guessing PDF URLs. The
`discover` command queries provider APIs (arXiv, Crossref, Semantic Scholar today; the registry in
`src/research_kb/discovery/` takes more) and merges `{filename, title, url}` entries into the
manifest `acquire` reads. Discovery only *finds* candidates — `acquire` still downloads and verifies
every one (HTTP 200 + `%PDF` magic + ≥60% title overlap), so a wrong URL is rejected, never indexed.

```bash
uv run research-kb discover "your topic terms" -p arxiv -p crossref   # → work/acquire-manifest.yaml
uv run research-kb acquire --manifest work/acquire-manifest.yaml       # download + verify
uv run research-kb index --scan reference
```

For the incremental-refresh cron (playbook step 9), `--refresh` fetches only material newer than each
provider's last run and advances the marker (persisted in `work/discovery-state.yaml`):

```bash
uv run research-kb discover "your topic terms" --refresh   # only the delta since last run
```

## Ingesting a web page

A remote page, standard, or blog post can be pulled straight into the corpus as Markdown — no PDF
required. `fetch` downloads the URL with the *same* believable-browser fingerprint as `acquire`,
extracts the **main content** (nav/boilerplate stripped, via trafilatura), content-verifies it (a
challenge/CAPTCHA/error/empty page is rejected, not written — there is no `%PDF` magic to lean on, so
verification requires a real title plus a non-trivial body), and writes an ordinary `.md` into
`reference/`. It is then indexed like any local Markdown, with no special-casing downstream.

```bash
uv run research-kb fetch https://example.org/some-standard   # → reference/<title-slug>.md
uv run research-kb index --scan reference                    # ingest it like any .md
```

## MCP tools

Run the server with `uv run research-kb-mcp` (the checked-in `.mcp.json` registers it for Claude
Code). It exposes five tools:

- `kb_search(query, k, doc_type, tier, phase, facets)` — hybrid search; `facets` is a name→value
  mapping over the profile's declared facet axes. Split a query on `|` to fuse independent search
  terms. Returns ranked hits with full provenance.
- `kb_get_paper(identifier)` — a document's metadata, section outline, and distilled-artifact paths.
- `kb_get_context(chunk_id)` — the parent section and neighbouring chunks around a hit.
- `kb_follow_citations(document_id, direction)` — walk the citation graph (`out` = works this cites;
  `in` = corpus works that cite it).
- `kb_list_corpus(tier, doc_type, phase, include_acquisition)` — what's indexed, optionally with the
  ranked cited-but-missing acquisition targets.

## Data layout

- `reference/` — the corpus. The only directory scanned. PDFs plus `.md`/`.txt` text sources.
- `work/` — all server state: `work/data/kb.sqlite` (the index), `work/distilled/<stem>/` (per-source
  artifacts: `verbatim.md`, `llm.md`, `enriched.md`, `divergence-report.md`), `work/eval/` (gold set).
- `src/research_kb/` — the engine.

## Configuration

Every knob is an environment variable with the `KB_` prefix (see `src/research_kb/config.py`):

| Variable | Default | Meaning |
| --- | --- | --- |
| `KB_EMBED_BACKEND` | `fastembed` | `fastembed` (in-process ONNX), `hashing` (offline), or `tei` (HTTP) |
| `KB_EMBED_MODEL` | `BAAI/bge-base-en-v1.5` | embedding model (ignored by the hashing backend) |
| `KB_EMBED_DIM` | `768` | embedding dimension; the DB records it and refuses a mismatch |
| `KB_CHUNK_TARGET_TOKENS` | `512` | target chunk size (atomic units may exceed it) |
| `KB_SEMANTIC_WEIGHT` / `KB_KEYWORD_WEIGHT` | `0.7` / `0.3` | RRF fusion weights |
| `KB_RRF_K` | `60` | RRF rank constant |
| `KB_PDF_EXTRACTOR` | `pymupdf` | `pymupdf` (fast) or `marker` (heavier, higher-fidelity layout) |
| `KB_LLM_EXTRACT_BACKEND` | `claude_cli` | distill backend for the LLM cross-check — see [Distill backends](#distill-backends) |
| `KB_BACKUP_ARTIFACTS_TARGETS` | `["rclone"]` | target list for the distillation artifacts (cloud by default) — see [Backup & restore](#backup--restore) |
| `KB_BACKUP_INDEX_TARGETS` | `["none"]` | target list for the index ("don't backup" — `rebuild` is its safety net) |
| `KB_BACKUP_ENCRYPT` | `true` | client-side encryption (rclone `crypt`) on for every target; opt out per target with `KB_BACKUP_PLAINTEXT_TARGETS` |
| `KB_BACKUP_LOCAL_DIR` | _(unset)_ | the `local` target's directory / mounted drive |
| `KB_BACKUP_RCLONE_REMOTE` | _(unset)_ | the `rclone` target's remote, e.g. `gdrive:kb-backup` (set up out-of-band) |
| `KB_BACKUP_PASSPHRASE` | _(unset)_ | crypt passphrase fallback when the OS keychain is unavailable (never commit it) |

A KB is self-describing: the embedding backend/model/dimension are written into the DB at index
time, so queries reproduce the index's vector space without re-supplying `KB_EMBED_*`.

## Distill backends

The "distill" step (the independent second-opinion transcription that feeds the divergence
cross-check) runs through a small **backend registry** in
[`src/research_kb/extract/backends.py`](src/research_kb/extract/backends.py). `KB_LLM_EXTRACT_BACKEND`
selects one by name:

| Backend | What it is | Index-time |
| --- | --- | --- |
| `claude_cli` (default) | headless `claude -p` under a Claude subscription (no API billing); segments the PDF and stitches | no — run via `distill` |
| `api` | the Anthropic API, single-call (bills per token; needs `ANTHROPIC_API_KEY`) | yes |
| `none` | deterministic-only; skip the cross-check | yes (no-op) |

Only Claude ships today. `codex_cli` and `opencode_cli` are **reserved names** for the same contract.

**Adding a backend** (e.g. Codex or OpenCode) is three steps — no dispatcher to edit:

1. Write a sibling module (like `extract/claude_cli.py`) exposing an `available(settings) -> bool` and
   an `extract(path, settings) -> str | None` that transcribes a PDF to Markdown.
2. In `extract/backends.py`, add one `register(Backend("codex_cli", …))` call, setting
   `index_time_safe` (leave `False` for a heavy CLI so `index` never auto-spawns it) and an
   `unavailable_hint`.
3. Give the backend its own config knobs on `Settings` (mirroring the `claude_*` fields); the shared
   `KB_LLM_EXTRACT_BACKEND` just picks which backend runs.

`extract.llm.run_live_extraction` and the `distill` command resolve the backend through the registry,
so nothing else changes.

## Backup & restore

Two assets are irreplaceable, and are protected differently:

- **Distillation artifacts** (`work/distilled/<stem>/*.md`) — hours of token-expensive LLM
  transcription. Default target: **a cloud provider** (via rclone).
- **The index** (`work/data/kb.sqlite`) — rebuildable, >100 MB, gitignored. Default target: **`none`
  ("don't backup")**; [`research-kb rebuild`](#rebuild) reconstructs it in one command.

Backup is **explicit** — it runs only on `research-kb backup push` (or the refresh cron), never during
`index`/`distill`, so serving stays offline. With nothing configured, `push`/`restore` are a clear
no-op pointing at the setup wizard; the KB ships fully queryable without any backup.

**First-time setup is `research-kb backup init`** — an interactive wizard that detects rclone and the
OS keychain, lets you pick each asset's targets (cloud / local / don't-backup; artifacts default to a
cloud provider, the index to "don't backup"), generates the crypt passphrase and **shows it once** to
save (caching it in the keychain), proves each target with a real push → pull → verify round-trip
(refusing to save a target that fails), and writes `KB_BACKUP_*` to `work/backup.env` so later
`push` / the refresh cron run **headless** — no rerun. Cloud auth stays with you (`rclone config`); the
wizard never handles provider secrets.

```bash
research-kb backup init                                  # one-time interactive setup (targets, key, round-trip)
research-kb backup push [--asset artifacts|index|all]   # encrypt + upload; each upload is verified
research-kb backup restore [--asset …] [--target …]     # download + decrypt + verify into place
research-kb backup status                                # targets, last-push time, drift
```

**Targets are a registry** ([`src/research_kb/backup/targets.py`](src/research_kb/backup/targets.py)),
the same shape as the distill-backend registry — adding one (WebDAV, IPFS, …) means registering a
module, not editing a dispatcher:

| Target | What it is |
| --- | --- |
| `rclone` | any configured rclone remote — one adapter, every provider (Drive, Dropbox, OneDrive, Proton, S3, …); chunked, resumable, checksummed |
| `local` | a directory / mounted drive (the on-site copy of a 3-2-1 setup, and the test target) |
| `none` | "don't backup" (a valid, default choice for the index) |

Each asset's target is a **list**, so cloud **+** local is possible. A `work/backup-manifest.json`
records every file's SHA-256 + where it was pushed + when; restore fetches by the manifest and
**re-verifies each hash** — a backup that can't be proven to restore is not a backup.

**Encryption is on by default for every target** (including `local`), via rclone's `crypt` overlay.
The passphrase is read non-interactively from the **OS keychain** (`keyring`), falling back to the
`KB_BACKUP_PASSPHRASE` env var, then an interactive prompt — it is **never** written to the repo or
the cloud. Generating the passphrase and caching it in the keychain is the **backup-setup wizard**'s
job; the engine only consumes it.

**Restoring on a fresh machine:** (1) install rclone and re-auth the remote (`rclone config` / the
provider's OAuth — the same account); (2) supply the crypt passphrase (from your password manager into
the keychain, or `KB_BACKUP_PASSPHRASE`); (3) `research-kb backup restore` to pull the artifacts back;
(4) `research-kb rebuild` if the index was not backed up.

### rebuild

```bash
research-kb rebuild    # reconstruct work/data/kb.sqlite from reference/ + committed artifacts
```

`rebuild` deletes any existing index and re-runs the pipeline, **reusing** the committed
`distilled/<stem>/llm.md` transcriptions (it never re-distills — the heavy `claude_cli` backend is not
index-time-safe), so a fresh clone with the artifacts present becomes queryable in one command. This
is why the index can safely default to "don't backup".

## Registering the server

Claude Code picks up the checked-in `.mcp.json` inside this repo. From another project:

```bash
claude mcp add --scope user research-kb -- uv run --directory /path/to/this/repo research-kb-mcp
```

**OpenCode** — add to `opencode.json` (a local, stdio server is a `command` array):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "research-kb": {
      "type": "local",
      "command": ["uv", "run", "--directory", "/path/to/this/repo", "research-kb-mcp"],
      "enabled": true
    }
  }
}
```

**Codex** — add to `~/.codex/config.toml` (or run `codex mcp add research-kb -- uv run --directory /path/to/this/repo research-kb-mcp`):

```toml
[mcp_servers.research-kb]
command = "uv"
args = ["run", "--directory", "/path/to/this/repo", "research-kb-mcp"]
```

The server is transport-agnostic stdio, so any MCP client registers it the same way — point the
client at `uv run --directory <repo> research-kb-mcp`.
