"""Runtime configuration, resolved from ``KB_*`` environment variables.

Every swappable knob (embedding backend/dimension, chunk sizes, RRF weights, tier
membership) lives here so the eval harness can drive it. Defaults are chosen so the
whole stack runs offline with no external services.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = two levels up from this file (src/research_kb/config.py -> repo).
REPO_ROOT = Path(__file__).resolve().parents[2]

# Core-tier sources: the load-bearing works an agent may quote directly. Matched against the
# source file stem. Empty by default — an instance fills it with its own core works.
DEFAULT_CORE_SOURCES: frozenset[str] = frozenset()


class Settings(BaseSettings):
    """Environment-driven settings. Prefix ``KB_`` (e.g. ``KB_EMBED_BACKEND=tei``)."""

    model_config = SettingsConfigDict(env_prefix="KB_", extra="ignore")

    # --- Storage -----------------------------------------------------------------
    # All MCP-server state lives under work/ (index, distilled artifacts, eval set).
    db_path: Path = Field(default=REPO_ROOT / "work" / "data" / "kb.sqlite")
    distilled_dir: Path = Field(default=REPO_ROOT / "work" / "distilled")

    # --- Corpus scan root --------------------------------------------------------
    # source_path is stored relative to base_dir (stable document identity across machines).
    # The corpus is exactly `reference/` (external material). `docs/` (the project's own research)
    # and `work/` (server state) are never scanned; `docs_dir` is kept only so tooling can locate
    # the project's writing, not as a scan root.
    base_dir: Path = Field(default=REPO_ROOT)
    reference_dir: Path = Field(default=REPO_ROOT / "reference")
    docs_dir: Path = Field(default=REPO_ROOT / "docs")

    # --- Embedding ---------------------------------------------------------------
    # 'fastembed' = in-process neural ONNX (bge-base, a core dependency) — the DEFAULT,
    # and it complains loudly rather than silently degrading if the package or model is missing;
    # 'hashing' = deterministic offline fallback; 'tei' = HTTP text-embeddings-inference. Only
    # *indexing* reads this — queries follow the backend recorded in the DB meta (self-describing KB).
    embed_backend: str = Field(default="fastembed")
    embed_url: str = Field(default="http://localhost:8080")
    embed_model: str = Field(default="BAAI/bge-base-en-v1.5")  # 768-dim; ignored by the hashing backend
    embed_dim: int = Field(default=768)
    embed_batch_size: int = Field(default=64)

    # --- LLM (extraction + enrichment) ------------------------------------------
    anthropic_model: str = Field(default="claude-opus-4-8")
    llm_max_tokens: int = Field(default=16000)

    # Live LLM-extraction ("distill") backend, resolved by name from research_kb.extract.backends
    # (used only when no committed llm.md exists):
    #   'claude_cli' — headless `claude -p` under a Claude subscription (no API billing, the default);
    #   'api'        — Anthropic API (bills per token, needs ANTHROPIC_API_KEY);
    #   'none'       — deterministic-only; skip the LLM cross-check.
    # Reserved for future backends with the same contract (not yet implemented): 'codex_cli', 'opencode_cli'.
    # Each backend owns its own knobs; the claude_cli backend reads the claude_* fields below.
    llm_extract_backend: str = Field(default="claude_cli")
    claude_bin: str = Field(default="claude")
    claude_model: str = Field(default="sonnet")  # claude-sonnet-5 — project default for LLM passes
    claude_effort: str = Field(default="high")
    # `claude -p` transcription is segmented by page range and stitched together: a whole-document
    # transcription overflows the output ceiling, and long dense segments also trip the *output
    # content filter* (a 400 on ~8 dense pages of prose). 4 pages/segment stays clear of the
    # filter; a segment that still trips it is bisected automatically (extract/claude_cli.py), so this
    # is a starting size, not a hard limit.
    llm_segment_pages: int = Field(default=4)
    claude_timeout_s: int = Field(default=900)
    claude_max_retries: int = Field(default=2)

    # --- Chunking ----------------------------------------------------------------
    chunk_target_tokens: int = Field(default=512)
    chunk_overlap_tokens: int = Field(default=50)
    chunk_hard_max_tokens: int = Field(default=1024)  # non-atomic chunks are split above this

    # --- Hybrid search -----------------------------------------------------------
    semantic_weight: float = Field(default=0.7)
    keyword_weight: float = Field(default=0.3)
    rrf_k: int = Field(default=60)

    # --- Extraction backend ------------------------------------------------------
    # 'pymupdf' (default, fast) or 'marker' (heavy, higher-fidelity layout).
    pdf_extractor: str = Field(default="pymupdf")

    core_sources: frozenset[str] = Field(default=DEFAULT_CORE_SOURCES)

    @property
    def has_anthropic(self) -> bool:
        """True when an Anthropic key is present, enabling LLM extraction/enrichment."""
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def artifact_dir(self, doc_stem: str) -> Path:
        """Directory holding distillation artifacts for a document stem."""
        return self.distilled_dir / doc_stem


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()
