# research-mcp

A Claude Code **plugin** that bootstraps a fidelity-first research knowledge base (RAG) on *any*
topic and serves it to agents over the Model Context Protocol.

It is a **template + bootstrap playbook**, not a knowledge base itself. Installed once per user or
org, it ships:

- a corpus-agnostic retrieval **engine** in [`template/`](template/) (PDF/text extraction, an
  optional independent LLM transcription with a divergence cross-check, structural chunking,
  in-process ONNX embeddings, `sqlite-vec` + FTS5 hybrid search fused with Reciprocal Rank Fusion,
  and five MCP tools), and
- a `/build-kb` command that drives the [`AGENTS.md`](AGENTS.md) playbook to stand up a
  **project-local knowledge base instance**: a clone of `template/` with its own corpus, index, and
  domain profile, registered as that project's MCP server.

One plugin, many project-local instances — each self-contained.

## Install

**Try it locally (no install):**

```bash
claude --plugin-dir /path/to/research-mcp
```

**Validate the plugin** before installing or distributing:

```bash
claude plugin validate /path/to/research-mcp
```

Once loaded, the `/research-mcp:build-kb` command is available (see the entry point under
[`skills/`](skills/)). For public distribution, submit the repository to the Claude Code plugin
directory.

## Usage

From inside the project that will use the KB:

```
/build-kb <topic>
```

The agent then follows the playbook in [`AGENTS.md`](AGENTS.md) end to end:

0. **Scope** the topic with you (subtopics, seed sources, providers, size, language, quoting policy,
   a name for the KB).
1. **Clone** the engine from `${CLAUDE_PLUGIN_ROOT}/template/` into `<project>/<name>-kb/` and rename
   the package.
2. **Author the domain profile** — the handful of tuned knobs (facets, core sources, claim markers,
   atomic units, extraction prompt, embedding model).
3. **Discover sources** into a *verified* manifest via provider APIs · **checkpoint**.
4. **Stand up** the server (`uv sync`, `init`, register MCP).
5. **Seed** the corpus (`acquire` + `index`) · **checkpoint**.
6. **Distill** the core sources and clear the divergence/validation gate.
7. **Evaluate** against gold queries (recall@k / MRR / faithfulness).
8. **Expand** via the citation graph.
9. **Schedule** an incremental refresh.
10. **Hand off** with the instance's own `AGENTS.md` and a tutorial.

## Layout

| Path | What it is |
|------|-----------|
| [`AGENTS.md`](AGENTS.md) | The bootstrap playbook + operating manual (the source of truth). |
| [`template/`](template/) | The generic engine, cloned per topic. See its `README.md` for the architecture and its `AGENTS.md` for query-time guidance. |
| `skills/build-kb/` | The `/build-kb` entry point. |
| `.claude-plugin/plugin.json` | The plugin manifest. |

## Requirements

- [`uv`](https://docs.astral.sh/uv/) for the engine's Python environment.
- `claude` on `PATH` for the optional distillation pass (headless `claude -p` under a Claude
  subscription — see the "distill backend" section of [`template/README.md`](template/README.md)).
  The KB is fully queryable without it.

## License

BSD 3-Clause — see [`LICENSE`](LICENSE). Copyright (c) 2026 CjS77. Redistributions (including cloned
KB instances) must retain the copyright notice, and the author's name may not be used to endorse
derivative works without permission.
