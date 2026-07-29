"""LLM extraction — the independent second pass.

An LLM ingests the PDF natively and returns Markdown. This is *never* trusted alone; its only role
is the second opinion in the divergence cross-check. The layer degrades gracefully:

1. a committed ``distilled/<stem>/llm.md`` artifact is used if present (reproducible, reviewable);
2. otherwise, at ``index`` time, only the cheap single-call ``api`` backend may run live — the heavy
   segmented ``claude_cli`` backend is produced deliberately via ``research-kb distill`` (see below);
3. otherwise it returns ``None`` and the pipeline runs deterministic-only.

The live backend is selected by ``KB_LLM_EXTRACT_BACKEND`` and resolved through the registry in
:mod:`research_kb.extract.backends` (``claude_cli`` — the default, ``api``, ``none``; ``codex_cli`` /
``opencode_cli`` reserved). Adding a backend means registering a module there, not editing the
dispatch here.
"""

from __future__ import annotations

import base64
from pathlib import Path

from ..config import Settings, get_settings
from .backends import get_backend

_EXTRACTION_PROMPT = (
    "Transcribe this PDF into faithful GitHub-flavored Markdown. Preserve structure exactly: "
    "headings and their numbering, tables, lists, and code blocks. Preserve any equations as LaTeX "
    "($...$ inline, $$...$$ display). Do NOT summarize, correct, or reword — transcribe exactly what "
    "the document says. Output only the Markdown."
)


def load_llm_artifact(doc_stem: str, settings: Settings | None = None) -> str | None:
    """Return a committed ``llm.md`` transcription for this document, if one exists."""
    settings = settings or get_settings()
    artifact = settings.artifact_dir(doc_stem) / "llm.md"
    return artifact.read_text(encoding="utf-8") if artifact.exists() else None


def run_api_extraction(path: Path, settings: Settings | None = None) -> str | None:
    """Call the Anthropic API on the PDF natively (bills per token). Returns Markdown, or None."""
    settings = settings or get_settings()
    if not settings.has_anthropic:
        return None
    try:
        import anthropic
    except ImportError:
        return None
    try:
        client = anthropic.Anthropic()
        data = base64.standard_b64encode(path.read_bytes()).decode()
        message = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=settings.llm_max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {"type": "base64", "media_type": "application/pdf", "data": data},
                        },
                        {"type": "text", "text": _EXTRACTION_PROMPT},
                    ],
                }
            ],
        )
        return "".join(block.text for block in message.content if getattr(block, "type", None) == "text")
    except Exception:
        return None


def run_live_extraction(path: Path, settings: Settings | None = None) -> str | None:
    """Run the configured live distill backend (see :mod:`research_kb.extract.backends`)."""
    settings = settings or get_settings()
    backend = get_backend(settings.llm_extract_backend)
    return backend.extract(path, settings) if backend else None


def get_llm_markdown(path: Path, doc_stem: str, settings: Settings | None = None) -> str | None:
    """Index-time entry point: committed artifact first, then only an *index-time-safe* backend.

    The heavy CLI backends are never auto-spawned here (they would fan out hundreds of subprocesses
    across a corpus re-index); they are produced ahead of time by ``research-kb distill``. Only a
    backend that marks itself ``index_time_safe`` (e.g. the single-call ``api`` backend) runs live.
    """
    settings = settings or get_settings()
    artifact = load_llm_artifact(doc_stem, settings)
    if artifact is not None:
        return artifact
    backend = get_backend(settings.llm_extract_backend)
    if backend is not None and backend.index_time_safe:
        return backend.extract(path, settings)
    return None


def generate_llm_artifact(
    path: Path, doc_stem: str, settings: Settings | None = None, force: bool = False
) -> Path | None:
    """Produce ``distilled/<stem>/llm.md`` via the configured live backend; return its path (or None).

    This is where the expensive transcription runs — once, committed and reviewable — so that
    ``index`` only ever reads the artifact. Driven by ``research-kb distill``.
    """
    settings = settings or get_settings()
    out_path = settings.artifact_dir(doc_stem) / "llm.md"
    if out_path.exists() and not force:
        return out_path
    markdown = run_live_extraction(path, settings)
    if markdown is None:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    return out_path
