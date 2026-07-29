"""Distill (LLM-extraction) backends and their registry.

A *distill backend* produces the independent, second-opinion transcription of a source document —
the input to the divergence cross-check (:mod:`research_kb.divergence`). It is deliberately never
trusted alone; its only role is to disagree with the deterministic extractor so a human can adjudicate.

Backends are looked up by name (the ``KB_LLM_EXTRACT_BACKEND`` setting), so adding one — e.g. a Codex
or OpenCode CLI — means *registering a module here*, not editing a dispatcher. Each backend supplies:

- ``available(settings) -> bool`` — can it run right now? (its CLI on PATH, an API key present, …)
- ``extract(path, settings) -> str | None`` — transcribe the document to Markdown, or ``None``.
- ``index_time_safe`` — cheap enough for ``index`` to invoke live? The heavy CLI backends are not
  (``index`` would fan out hundreds of subprocesses); they are produced ahead of time by
  ``research-kb distill``. Only cheap single-call backends set this ``True``.
- ``unavailable_hint(settings) -> str`` — a one-line fix shown when ``available`` is ``False``.
- ``complete(prompt, settings) -> str | None`` — *optional* plain text-in/text-out generation (no
  PDF), reusing the same model access as distillation for authoring tasks such as
  ``research-kb profile-init``. ``None`` when a backend offers no text path (e.g. the ``none`` backend).

Only the Claude backend ships today. ``codex_cli`` and ``opencode_cli`` are reserved names for the
same contract — implement them by adding a sibling module and one ``register(...)`` call below.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..config import Settings


def _default_hint(settings: Settings) -> str:
    return "backend unavailable"


@dataclass(frozen=True)
class Backend:
    """One distill backend: a named pair of ``available`` / ``extract`` callables."""

    name: str
    available: Callable[[Settings], bool]
    extract: Callable[[Path, Settings], str | None]
    index_time_safe: bool = False
    unavailable_hint: Callable[[Settings], str] = _default_hint
    # Optional text-in/text-out generation (no PDF). Present on backends whose model access can serve
    # authoring tasks (profile-init); ``None`` when the backend has no such path.
    complete: Callable[[str, Settings], str | None] | None = None


_REGISTRY: dict[str, Backend] = {}


def register(backend: Backend) -> None:
    """Add (or replace) a backend in the registry."""
    _REGISTRY[backend.name] = backend


def get_backend(name: str) -> Backend | None:
    """Look up a backend by name; ``None`` if the name is unknown."""
    return _REGISTRY.get(name)


def backend_names() -> list[str]:
    """Registered backend names, for help text and diagnostics."""
    return sorted(_REGISTRY)


# Backends import their implementation lazily (inside the wrappers) so registering this module never
# pulls in pymupdf / anthropic at import time, and there is no import cycle with .llm / .claude_cli.

# --- claude_cli: headless `claude -p`, subscription auth (default; heavy — produced via `distill`) --


def _claude_available(settings: Settings) -> bool:
    from .claude_cli import claude_available

    return claude_available(settings)


def _claude_extract(path: Path, settings: Settings) -> str | None:
    from .claude_cli import run_claude_extraction

    return run_claude_extraction(path, settings)


def _claude_hint(settings: Settings) -> str:
    return f"`{settings.claude_bin}` not found on PATH; install Claude Code or set KB_CLAUDE_BIN"


def _claude_complete(prompt: str, settings: Settings) -> str | None:
    from .claude_cli import run_claude_prompt

    return run_claude_prompt(prompt, settings)


# --- api: Anthropic API, single-call (cheap — safe to run live at index time) --------------------


def _api_available(settings: Settings) -> bool:
    return settings.has_anthropic


def _api_extract(path: Path, settings: Settings) -> str | None:
    from .llm import run_api_extraction

    return run_api_extraction(path, settings)


def _api_complete(prompt: str, settings: Settings) -> str | None:
    from .llm import run_api_prompt

    return run_api_prompt(prompt, settings)


def _api_hint(settings: Settings) -> str:
    return "set ANTHROPIC_API_KEY and install the 'llm' extra (anthropic)"


register(
    Backend(
        "claude_cli", _claude_available, _claude_extract,
        index_time_safe=False, unavailable_hint=_claude_hint, complete=_claude_complete,
    )
)
register(
    Backend("api", _api_available, _api_extract, index_time_safe=True, unavailable_hint=_api_hint, complete=_api_complete)
)
register(Backend("none", lambda s: True, lambda p, s: None, index_time_safe=True))
