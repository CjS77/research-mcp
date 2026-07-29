# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-07-29

### Added

- Named N-ary facets replacing the fixed `facet_a`/`facet_b` columns, letting a topic declare as
  many filter axes as it warrants.
- Embedding-model advisor that suggests a model from the corpus, plus `eval` support for A/B
  comparing two embedding models before committing.
- Shared discovery library: a provider registry with arXiv, Crossref, and Semantic Scholar clients.
- IACR ePrint and Europe PMC discovery providers.
- `research-kb discover` CLI with offline discovery tests.
- Fetch a remote page into the corpus as Markdown (web-source ingestion).
- `profile-init` step that drafts a domain profile from a topic description.
- Backup and artifact preservation (K-6): a backup init wizard and fresh-machine recovery flow.

### Documentation

- Describe named facets in the playbook and query-time guides.
- Document the `research-kb discover` command in the engine README.
- Reference `profile-init` in the playbook and engine README.
- Document the backup init wizard and fresh-machine recovery.

### Tests

- Cover `acquire` backoff and non-PDF rejection paths.
- Offline coverage for web-source ingestion.
- Offline coverage for the backup-setup wizard flow.

### Chores

- Fix unit tests.
- Ignore the kanban board.
- Run CI on pull requests only.

## [0.1.0]

### Added

- Initial `research-mcp` KB template plugin: the corpus-agnostic engine (extract, cross-check,
  chunk, embed, hybrid search) and the `/build-kb` bootstrap playbook.
- License.

[0.2.0]: https://github.com/CjS77/research-mcp/releases/tag/v0.2.0
[0.1.0]: https://github.com/CjS77/research-mcp/releases/tag/v0.1.0
