# <KB-NAME> — agent guide

This repo serves a `<KB-NAME>` knowledge base over MCP. When researching topics this KB covers,
query it instead of grepping `reference/` or reading source files directly.

> Replace `<KB-NAME>` and the description above with this instance's name and scope, and note the two
> facet axes (`facet_a`, `facet_b`) this corpus filters on.

## Invoking

Use the `research-kb` MCP server's tools (registration below; the `research-kb` CLI mirrors every
tool if the server isn't loaded). Start with `kb_search`:

```
kb_search(query="<query>", k=10)
```

- Split a query into independent facets with `|` — each facet is embedded separately and the
  result lists are fused: `kb_search(query="index structure | cache eviction | tail latency")`.
- Phrase queries as concepts, but keep exact technical tokens verbatim. The keyword half of the
  search leans on those tokens, so spelling them exactly matters.
- Narrow the corpus with the optional parameters: `tier` (`core`|`breadth`), `doc_type`
  (`paper`|`research`|`assessment`|`sketch`|`spec`), `facet_a`, `facet_b`, `phase` (`1`|`2`|`3`).
  e.g. `tier="core"` restricts to the handful of validated core sources.

Follow-on tools (a search hit hands you the `chunk_id` / `document_id` these need):

- `kb_get_context(chunk_id=<id>)` — parent section + neighbouring chunks around a hit; the first
  reach for reading past the 400-char snippet.
- `kb_get_paper(identifier="<id-or-title>")` — metadata, section outline, and paths to the
  distilled artifacts. The distilled `work/distilled/<stem>/llm.md` is the best whole-document read.
- `kb_follow_citations(document_id=<id>, direction="out"|"in")` — walk the citation graph
  (`out` = works this document cites; `in` = corpus documents that cite it).
- `kb_list_corpus(tier=..., include_acquisition=true)` — see what is indexed, optionally with the
  ranked cited-but-missing works to acquire next.

## Interpreting results

Each hit prints as `[score] <title>  §<section> p<page>  (<content_kind>, chunk <id>)`
followed by a snippet.

- **The score is a relative rank, not a similarity or confidence.** It is an RRF fusion score
  (semantic weight 0.7 + keyword 0.3, `rrf_k=60`), so absolute values are small (~0.01–0.02) and
  only comparable *within a single result set*. Read the top-to-bottom ordering; never gate on the
  absolute number and never compare scores across different queries.
- A hit points at a *passage*, not a whole document. `§section` + `p<page>` is your citation; the
  `chunk <id>` is the handle for expanding context.
- `content_kind` says what you're reading: `verbatim` is exact source text (safe to quote);
  `derived_summary` is a generated section summary (good for orientation — cite the underlying
  source, not the summary).
- The snippet is a 400-char preview. To read around a hit, use `kb_get_context` (parent section +
  neighbouring chunks) or open the document's distilled `llm.md`.
- No strong hit? Re-query with different facets or synonyms before falling back to grep — a
  vocabulary-dense corpus is sensitive to exact wording.

## CLI or MCP server?

Prefer the **MCP server** when it is loaded; fall back to the CLI otherwise. `research-kb` is
designed MCP-first (`src/research_kb/mcp_server.py` is the primary interface), and the server
exposes five tools — `kb_search`, `kb_get_paper`, `kb_get_context`, `kb_follow_citations`,
`kb_list_corpus` — that turn the natural search → expand → traverse loop into typed calls.
Advantages over the CLI: results return as structured JSON (no text parsing), filters are typed
parameters, and the process stays warm (each `uv run research-kb` invocation is a fresh Python +
embedder cold start). Use the CLI for one-off checks, when the server isn't registered, or when you
want a reproducible command in the transcript.

## Registering the server

Claude Code picks up the checked-in `.mcp.json` inside this repo. From other projects:

```bash
claude mcp add --scope user research-kb -- uv run --directory /path/to/this/repo research-kb-mcp
```

OpenCode (`opencode.json`) and Codex (`~/.codex/config.toml`) snippets are in
[README.md](README.md#registering-the-server).
