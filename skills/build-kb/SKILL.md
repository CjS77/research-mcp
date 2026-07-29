---
name: build-kb
description: Bootstrap a fidelity-first research knowledge base (RAG, served over MCP) on any topic.
argument-hint: <topic>
disable-model-invocation: true
---

The user wants to build a research knowledge base on: **$ARGUMENTS**

First read the full bootstrap playbook at `${CLAUDE_PLUGIN_ROOT}/AGENTS.md` and follow it end to end.
That file is the source of truth — the CLI commands, the domain knobs and their locations, the
source-discovery reference, and the fidelity invariants all live there. In brief:

0. **Scope it.** Interview the user for subtopics/seed terms, seed sources, providers & source
   types, time range, size, language, and how much the agent may quote directly. Ask what they want
   to **name** the KB (default `$ARGUMENTS-kb`, e.g. `dinosaurs-kb`).
1. **Instantiate** — clone `${CLAUDE_PLUGIN_ROOT}/template/` into the named instance dir (inside the
   project that will use the KB) and rename the package.
2. **Author the domain profile** — the ~9 tuned knobs (facets, core set, claim markers, atomic
   units, notation, extraction prompt, embedding model).
3. **Discover sources** → a *verified* manifest via provider APIs. **Checkpoint:** show the user the
   manifest before bulk download.
4. **Stand up** the server (`uv sync`, `init`, register MCP, smoke-test).
5. **Seed** — `acquire` (verified) then `index`. **Checkpoint:** show a sample + real searches.
6. **Distill** the core/high-value sources; clear the divergence/validation gate.
7. **Evaluate** — author gold queries, run `eval`, record the baseline.
8. **Expand** via the citation-graph acquisition queue.
9. **Schedule** an incremental refresh (ask cadence; set up cron).
10. **Hand off** — write the instance's own AGENTS.md + a query tutorial.

Do not skip the two checkpoints. Prefer the eval harness over ad-hoc queries as the "is it working"
gate. Keep every fidelity invariant in `${CLAUDE_PLUGIN_ROOT}/AGENTS.md` intact.
