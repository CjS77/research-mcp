# research-mcp — build a fidelity-first research KB on any topic

This repo is a **template + bootstrap playbook** for standing up a retrieval knowledge base (RAG)
on *any* subject and serving it to agents over the Model Context Protocol. This file is the
operating manual for the bootstrapping agent: when a user runs `/build-kb <topic>`, follow the
playbook below.

The engine under [`template/`](template/) is **corpus-agnostic** — it cares nothing about any
particular field. Only a handful of knobs encode domain knowledge, and your job at bootstrap is to
author them for the topic. Everything else (extract → cross-check → chunk → embed → hybrid search →
MCP) is reused as-is.

---

## Mental model: engine · profile · instance

- **Engine** — the generic pipeline in `template/` (package `research_kb`). PDF/text extraction
  (pymupdf), an optional independent LLM transcription, a divergence cross-check between the two,
  structural type-aware chunking, in-process ONNX embeddings, `sqlite-vec` + FTS5 hybrid retrieval
  fused with Reciprocal Rank Fusion, and five MCP tools. None of it is domain-specific.
- **Profile** — the ~9 domain-tuned knobs (facet vocabularies, the "core / quotable" source set,
  claim markers, atomic-unit keywords, notation guidance, the extraction prompt, the embedding
  model). These are what you write for the topic. See [What must be domain-tuned](#what-must-be-domain-tuned).
- **Instance** — one topic's KB: a directory the user names (e.g. `dinosaurs-kb/`) holding its
  corpus (`reference/`), distillation artifacts and index (`work/`), MCP registration, and its own
  `AGENTS.md` telling the *querying* agent how to search it. Config resolves relative to the
  instance root, so an instance is self-contained.

**Deployment model:** this plugin is installed **once per user or org** (it is not itself a KB — it
ships the engine and this playbook). Each KB **instance is scoped to a project**: you run
`/build-kb` from within the project that will use the KB, and the instance is scaffolded there and
registered as that project's MCP server (project-scope `.mcp.json`). One plugin, many
project-local instances — each a self-contained clone of `template/` with its own corpus, index,
and profile. Instances are deliberately independent; they don't share a running engine.

**The insight that makes this a template:** the pipeline is fixed; the domain is a small config an
LLM can author because it already knows the field's vocabulary. You are that LLM.

---

## The `/build-kb <topic>` playbook

Work these steps in order. Stop at the two **checkpoints** and get the user's read before spending
tokens on bulk download or distillation.

### 0. Scope the topic (interview — don't skip)

Before touching code, pin the corpus down with the user. Ask for:

- **Subtopics / seed terms** — the query strings you'll hit provider APIs with.
- **Seed sources** — landmark works, authors, venues, or URLs they already trust.
- **Source providers & types** — arXiv, IACR ePrint, PubMed / Europe PMC, Semantic Scholar,
  OpenReview, Crossref/DOI, or plain web docs / standards / blogs (the engine ingests PDFs *and*
  `.md`/`.txt`).
- **Time range & size** — "everything since 2018", or a curated ~100.
- **Language** — drives the embedding-model choice (step 2).
- **How much the agent may quote directly** — decides which sources become the *core* tier.
- **A name for the KB** — default `<topic>-kb`; the user may pick anything (`dinosaurs-kb`).

Turn the answers into a short written scope note; the manifest in step 3 is built from it.

### 1. Instantiate the engine

The instance belongs to the **project** that will use it — scaffold it inside (or alongside) that
project, not inside this plugin. Clone the engine from the installed plugin's `template/` into the
instance directory the user named. `${CLAUDE_PLUGIN_ROOT}` resolves to this plugin's install
directory, so the source path works wherever the plugin is installed:

```bash
rsync -a --exclude work --exclude '.venv' --exclude '__pycache__' \
      "${CLAUDE_PLUGIN_ROOT}/template/" <project>/<name>-kb/
```

Then rename the package and entry points so multiple KBs coexist without colliding (`<pkg>` =
`<name>` with hyphens turned to underscores):

- `src/research_kb/` → `src/<pkg>_kb/`
- `pyproject.toml`: `name`, both `[project.scripts]` (`research-kb` → `<name>-kb`,
  `research-kb-mcp` → `<name>-kb-mcp`), and `[tool.hatch.build.targets.wheel].packages`.
- `mcp_server.py`: `FastMCP("research-kb")` → `FastMCP("<name>-kb")`.
- `.mcp.json`: the server key and command.

`config.py` derives every path from the repo root, so the DB, `reference/`, and `distilled/` follow
the new directory automatically — no path edits needed.

### 2. Author the domain profile

Edit the knobs in [the table below](#what-must-be-domain-tuned). Concretely:

- Pick the **core set** — the load-bearing sources an agent may quote directly. Start empty; fill it
  after discovery. (`config.py` `DEFAULT_CORE_SOURCES`.)
- Fill the **two facet vocabularies** (`enrich.py` `_FACET_A_TERMS` / `_FACET_B_TERMS`) with the two
  most useful axes of *this* field — for dinosaurs maybe *clade* and *period*; for a drug field
  *compound* and *condition*; for a framework *module* and *version*. Optionally rename the
  `kb_search` params `facet_a` / `facet_b` to those words. These become the filterable facets.
- Rewrite the **claim markers** (`divergence.py`) and **atomic-unit keywords** (`chunk.py`) only if
  the generic defaults miss what "a load-bearing claim" or "an indivisible unit" means here.
- Set the **notation note** and **extraction prompt** for the medium (equations matter for STEM;
  leave the note empty for prose fields).
- Choose the **embedding model** (`KB_EMBED_MODEL`): the default `bge-base-en-v1.5` (general
  English) is fine for most; prefer a scientific model for dense academic text, or a multilingual
  model for non-English corpora. The DB records the backend in `kb_meta`, so this is fixed per KB at
  index time.

### 3. Discover sources → verified manifest · **Checkpoint 1**

Turning a topic into *verified, downloadable* URLs is the real work here — guessed URLs are usually
wrong, which is exactly why `acquire` verifies. For each provider (see
[Source discovery](#source-discovery-reference)):

1. Query the provider's **API** with the seed terms to get candidate `{title, id, url}` triples —
   never hand-guess PDF URLs.
2. Build `work/acquire-manifest.yaml` as a list of `{filename, title, url}`.
3. **Present the manifest to the user** — titles, counts, coverage of the subtopics — so they
   confirm you're going in the right direction *before* any bulk download. Adjust and re-query until
   they're happy.

### 4. Stand up the server

```bash
cd ../<name>-kb
uv sync                     # engine deps: pymupdf, sqlite-vec, fastembed, mcp, …
uv run <name>-kb init       # create work/data/kb.sqlite + schema
```

Smoke-test before seeding: `uv run <name>-kb status` (empty is fine), then index one document to
confirm the embedding model downloads and the pipeline runs end to end. Register the MCP server
(Claude Code auto-loads the checked-in `.mcp.json`; for user-wide use,
`claude mcp add --scope user <name>-kb -- uv run --directory <abs-path>/<name>-kb <name>-kb-mcp`;
OpenCode/Codex snippets are in the instance README).

### 5. Seed the corpus · **Checkpoint 2**

```bash
uv run <name>-kb acquire --manifest work/acquire-manifest.yaml --dest reference
uv run <name>-kb index --scan reference
```

`acquire` downloads each URL, checks PDF magic bytes, and verifies the first-page text overlaps the
expected title (≥60%) — a wrong or rotten URL is **rejected**, not indexed. `index` is incremental
(unchanged content hash skips) and runs fully offline with the deterministic extractor. Then
**present a sample to the user**: `uv run <name>-kb corpus` plus 3–5 real searches
(`uv run <name>-kb search "<question>" -k 8`). Confirm relevance and provenance look right.

### 6. Distill the core / high-value sources

This produces the independent LLM transcription that feeds the fidelity cross-check:

```bash
uv run <name>-kb distill --all              # or: distill <stem> … for specific docs
uv run <name>-kb index --scan reference     # re-index; now chunks the trustworthy transcription
```

`distill` runs the configured **distill backend** (`KB_LLM_EXTRACT_BACKEND`, resolved through
`extract/backends.py`). The default `claude_cli` uses headless `claude -p` under a Claude
subscription (no API billing), segmenting the PDF (~4 pages/segment) and stitching — so first check
`claude` is on `PATH`. (`api` uses the Anthropic API; `none` skips the cross-check; `codex_cli` /
`opencode_cli` are reserved for future backends — see "Adding a distill backend" in the engine
README.) It writes `work/distilled/<stem>/{verbatim,llm,enriched,divergence-report}.md`. Distillation
is **optional**: without it the KB serves fine deterministic-only. Reserve it for the sources that
matter most — start with the core tier.

**The validation gate:** a *core*-tier doc with an unresolved **semantic** divergence between the
two extractors is blocked from indexing until a human signs off. Surface any `BLOCKED` docs, read
the divergence report, and either fix the transcription or run
`uv run <name>-kb validate <source_path>` once you've confirmed the disagreement is benign.

### 7. Evaluate — an objective gate, not vibes

Use the real eval harness, not ad-hoc queries. Author `work/eval/gold_queries.yaml` (real questions
→ the `expected_source_path` that should answer each — copy the schema from
`examples/gold_queries.example.yaml` in the instance), then:

```bash
uv run <name>-kb eval -k 10         # recall@k / MRR / faithfulness, plus a list of misses
```

Record the baseline. **Tune knobs (chunk size, RRF weights, embedding model) only against this
number** — the eval arbitrates every change. Aim for >80% recall@10.

### 8. Expand via the citation graph (second seeding wave)

Once the seed corpus is indexed, the KB tells you what it's missing:

```bash
uv run <name>-kb corpus --acquire     # ranked cited-but-missing works
```

Add the high-frequency targets to the manifest, re-acquire, re-index. Repeat until coverage
plateaus. (For corpora without reference lists — docs, wikis — this wave is a no-op; skip it.)

### 9. Schedule refresh (cron)

Ask the user how often to check for new material, then schedule a job (harness `/schedule`, or a
system cron) that runs the **incremental refresh**:

1. Re-run **discovery** with a date filter (e.g. arXiv `submittedDate` since last run) → new
   candidates.
2. `acquire` them into `reference/` (verification rejects junk automatically).
3. `index --scan reference` — incremental; only new/changed docs cost anything.
4. `distill` any new core-tier docs; clear the validation gate.
5. `corpus --acquire` to refresh the citation-driven queue.
6. `eval` — catch retrieval regressions; alert the user if recall drops.

Persist a "last run" date so discovery only fetches the delta.

### 10. Hand off with a tutorial

- Write the instance's own **`AGENTS.md`** (query-time guidance): the facet names, how to phrase
  queries, how to read hits, the five tools. A generic version ships in the cloned engine — fill in
  the facets and corpus summary. Keep `CLAUDE.md` as `See @AGENTS.md`. This is what makes the
  *querying* agent effective; without it the facets are undiscoverable.
- Update the instance `README.md` with the corpus summary and registration commands.
- Give the user a short **tutorial**: a few worked `kb_search` calls, how to expand a hit with
  `kb_get_context`, how to walk citations, and how to grow the corpus (add to the manifest, re-run
  acquire + index).

---

## What must be domain-tuned

Everything else is reused unchanged. Locations are within `template/src/research_kb/`.

| Knob | Location | Default | Generalize to |
|------|----------|---------|---------------|
| Package / script / server names | `pyproject.toml`, `mcp_server.py` `FastMCP(...)`, `.mcp.json` | `research-kb` | `<name>-kb` |
| Core (quotable) source set | `config.py` `DEFAULT_CORE_SOURCES` / `core_sources` | empty | the load-bearing sources an agent may quote directly |
| Facet A / B vocabularies | `enrich.py` `_FACET_A_TERMS` / `_FACET_B_TERMS` | empty | the two most useful filter axes of the field |
| Facet names | `facet_a` / `facet_b` — params in `mcp_server.py` + `cli.py`, columns in `models.py`/`schema.sql` | generic | rename to the field's axes (clade/period, compound/condition, module/version…) |
| Claim markers | `divergence.py` `_CLAIM_MARKERS`, `_claim_severity` | generic strong-claim verbs & quantifiers | the words that mark a load-bearing claim in the field |
| Atomic-unit keywords | `chunk.py` `_ATOMIC_KEYWORDS` | academic (theorem/proof/algorithm/…) | the field's indivisible units (clause/holding, trial/cohort, listing/figure) |
| Notation note | `enrich.py` `notation_note` | none | domain notation guidance, or leave empty |
| Extraction prompt | `extract/llm.py` `_EXTRACTION_PROMPT` | faithful Markdown + LaTeX | domain-appropriate fidelity emphasis |
| Embedding model | `config.py` `embed_model` / `KB_EMBED_MODEL` | `bge-base-en-v1.5` (768-d, English) | scientific / multilingual / larger model per corpus & language |
| doc_type taxonomy | `models.py` `DocType`, `corpus.py` `_text_meta` | paper/research/assessment/sketch/spec | the document kinds that exist in the topic |

---

## Source discovery reference

Turn the scope note into a **verified** manifest. Query APIs; never hand-write PDF URLs.

| Provider | Discovery | PDF / verification notes |
|----------|-----------|--------------------------|
| **arXiv** | `export.arxiv.org/api/query` (Atom): category, keyword, `submittedDate` | `arxiv.org/pdf/<id>`. Reachable from most networks. |
| **IACR ePrint** | ePrint metadata API confirms `{id, title}` | Its PDF host Cloudflare-403s datacenter IPs; `acquire` already falls back to the Wayback mirror (`web.archive.org/web/2id_/<url>`). |
| **Semantic Scholar** | Graph API: search + citations + `openAccessPdf` | Great for the citation-graph expansion wave. |
| **PubMed / Europe PMC** | E-utilities / REST | Biomedical; Europe PMC exposes full-text PDFs. |
| **Crossref / DOI** | Metadata by DOI or title | Resolves publisher links; PDF availability varies. |
| **OpenReview** | Venue / forum API | ML conference papers + reviews. |
| **Generic web** | (roadmap) fetch → Markdown | The engine **already ingests local `.md`/`.txt`** as first-class corpus material — drop them in `reference/` and index. What's *not* yet built is fetching a remote page/doc and converting it to Markdown; until then, save the page as `.md`/`.txt` yourself. See roadmap. |

**Verification is non-negotiable.** `acquire` requires HTTP 200 + `%PDF` magic bytes + ≥60%
first-page/title word overlap. An LLM-guessed URL that points at the wrong document is rejected
rather than silently indexed. Present rejects to the user; don't quietly drop them.

**Download robustness (auto-download is flaky — harden `acquire`).** `acquire.py` today sends a
single, now-stale Chrome user-agent and only falls back to the Wayback mirror. Publishers and CDNs
increasingly block that fingerprint (this playbook's own source article 403'd a plain fetch). Bring
`acquire`'s HTTP up to a believable browser fingerprint:

- **Use current, real UA strings** (latest Chrome/Firefox on Windows/macOS) and **rotate** across a
  small pool per request. Never ship a library-default UA (`python-httpx/…`) or an outdated one —
  both are trivial bot tells.
- **Send the full header set that matches the UA**, so the fingerprint is internally consistent: for
  a Chrome UA include the client hints (`Sec-CH-UA`, `Sec-CH-UA-Mobile`,
  `Sec-CH-UA-Platform` matching the OS in the UA), the `Sec-Fetch-*` headers,
  `Accept`/`Accept-Language`/`Accept-Encoding`, and `Upgrade-Insecure-Requests`. A UA claiming
  Chrome with none of Chrome's other headers is the easiest block.
- **Set a plausible `Referer`** — the provider's abstract/listing page for the PDF you're fetching.
- **Reuse a session** (keep cookies the site sets) and **pace + back off**: jittered delays,
  exponential backoff on 429/5xx (already present for Wayback), and no bursts.
- Keep the mirror fallback, and keep verification unconditional — a friendlier fingerprint changes
  *whether you get the bytes*, never *whether you trust them*.

---

## Roadmap

The instance model is settled: this plugin installs once per user/org, and each KB is a
project-local clone of `template/` — see [Deployment model](#mental-model-engine--profile--instance).
Per-topic instances are the intended shape, **not** a limitation to engineer away. What's left is
making each instance better:

**Committed (build these):**

1. **Named, N-ary facets.** Generalize the two fixed facet columns (`facet_a`/`facet_b`) to a list
   of named facets declared in the profile, with the MCP tool exposing them dynamically. Removes the
   last hardcoded shape in the schema and lets a topic define as many filter axes as it warrants.
   This is the priority build.
2. **LLM-authored profile.** At bootstrap, have the agent draft the whole profile (facet
   vocabularies, atomic units, claim markers, extraction prompt) from the topic description — it
   knows the field. Ship a `profile-init` step that proposes a profile for the user to edit.
3. **Web-source ingestion.** Local `.md`/`.txt` are already first-class; add a fetch → Markdown
   adapter so a remote page/doc/standard can be pulled straight into `reference/`. Depends on the
   [download-robustness](#source-discovery-reference) hardening — reuse the same believable-browser
   fingerprint for page fetches, not just PDFs.
4. **Embedding-model advisor.** Suggest the embedding model from corpus language and density, and let
   `eval` A/B two models before committing (the DB is self-describing, so it's a rebuild, not a
   migration).
5. **Backup & artifact preservation — critical.** Two irreplaceable assets: the token-expensive
   **distillation artifacts** (`work/distilled/<stem>/{verbatim,llm,enriched,divergence-report}.md`
   — hours of LLM transcription you must never re-pay for) and the **index** (`work/data/kb.sqlite`,
   rebuildable from corpus + artifacts but slow). The index exceeds GitHub's 100 MB file limit and is
   `.gitignore`d; the artifacts should be committed *and* backed up. Provide push/restore to
   Google Drive, Proton Drive, and IPFS as backup targets, plus a one-command rebuild so a fresh
   clone is queryable fast. Losing distillation output is the expensive failure — protect it first.

**Deferred (keep on the roadmap, don't build now):**

6. **Shared discovery library.** Factor the provider API clients (arXiv/ePrint/S2/PMC/Crossref) into
   the engine so every instance gets the same verified-discovery + incremental-refresh machinery for
   free, instead of the bootstrapping agent reimplementing discovery each time.

---

## Invariants — do not break these

The engine's value is fidelity. Preserve these guarantees in every instance and every generalization:

- **Verbatim is sacred and separate from derived.** `content_kind` distinguishes exact source text
  (safe to quote) from generated summaries/enrichment. Never let derived content overwrite a
  verbatim span. Deterministic extraction is always written to `verbatim.md` as the baseline, even
  when a trustworthy LLM transcription is what gets chunked.
- **Provenance on every chunk.** `{document, section, page}` travels with each hit so an agent can
  cite precisely. Don't add a retrieval path that drops it.
- **Two independent extractors + a cross-check** beat trusting either. Keep the divergence gate for
  the core tier; semantic disagreements block until a human validates.
- **Tuning is eval-gated.** Chunk sizes, RRF weights, and the embedding model change only when the
  gold-query eval says they should.
- **Serving stays offline and private.** After the one-time embedding-model download, queries hit no
  external service. Indexing degrades gracefully: no `ANTHROPIC_API_KEY` / `claude` →
  deterministic-only; no GPU/TEI → in-process ONNX; the KB is always queryable.

---

## Reference

The engine lives in [`template/`](template/). Read its `README.md` for the architecture and its
`AGENTS.md` for query-time guidance. The `research-kb` CLI mirrors every MCP tool, which is the
fastest way to learn and test the pipeline. Users start a build with `/build-kb <topic>`.
