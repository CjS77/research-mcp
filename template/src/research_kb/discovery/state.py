"""Incremental-refresh bookkeeping: each provider's last-run date, persisted under ``work/``.

Step 9 of the playbook re-runs discovery on a schedule and wants only the *delta* — new material
since the previous run. This keeps a tiny ``{provider: last-run-date}`` map so a refresh can pass
each provider its own ``since`` and advance the marker afterwards. It is deliberately a plain YAML
file (human-inspectable, git-diffable), not a row in the index DB, so it survives an index rebuild.
"""

from __future__ import annotations

from datetime import date

import yaml

from ..config import Settings


def load_last_run(settings: Settings, provider: str) -> date | None:
    """The persisted last-run date for ``provider``, or ``None`` if it has never run."""
    path = settings.discovery_state_path
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = (data.get("last_run") or {}).get(provider)
    return date.fromisoformat(raw) if isinstance(raw, str) else None


def save_last_run(settings: Settings, provider: str, when: date) -> None:
    """Record ``when`` as ``provider``'s last-run date, preserving other providers' markers."""
    path = settings.discovery_state_path
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {} if path.exists() else {}
    last_run = dict(data.get("last_run") or {})
    last_run[provider] = when.isoformat()
    data["last_run"] = last_run
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=True, allow_unicode=True), encoding="utf-8")
