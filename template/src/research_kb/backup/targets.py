"""Backup targets and their registry — the same shape as the distill-backend registry.

A *backup target* is a named place an asset can be pushed to and pulled from. Targets are looked up
by name (the per-asset target lists in :class:`~research_kb.config.Settings`), so adding one — say an
IPFS or WebDAV target later — means *registering a module here*, not editing a dispatcher. Each
target supplies:

- ``available(settings) -> bool`` — is it configured enough to use right now? (a local dir set, an
  rclone remote named + the binary on PATH, …).
- ``push(transfer) -> str`` — upload the staged tree; return a human-readable location string.
- ``pull(transfer) -> None`` — download into the staging dir (the engine then verifies hashes).
- ``verify(settings) -> bool`` — best-effort reachability, for ``backup status``.
- ``unavailable_hint(settings) -> str`` — the one-line fix shown when ``available`` is ``False``.

Three targets ship: ``local`` (a directory / mounted drive — also the test target), ``rclone`` (any
rclone remote, i.e. every cloud provider), and ``none`` ("don't backup", the index default). Targets
import their implementation lazily inside the wrappers so importing this module pulls in nothing heavy
and there is no import cycle with the engine.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..config import Settings


@dataclass
class Transfer:
    """One asset moving to/from one target: the staged tree, plus the encryption decision + key.

    For a push the ``staging_dir`` is already populated with the asset's files (relative paths
    preserved); for a pull it is an empty temp dir the target fills. ``settings`` travels along so a
    target can read its own knobs (local dir, rclone remote) without a global lookup.
    """

    asset: str
    staging_dir: Path
    encrypted: bool
    passphrase: str | None
    settings: Settings


def _default_hint(settings: Settings) -> str:
    return "target unavailable — run the backup-setup wizard"


@dataclass(frozen=True)
class BackupTarget:
    """One backup target: a named bundle of the transfer + reachability callables."""

    name: str
    available: Callable[[Settings], bool]
    push: Callable[[Transfer], str]
    pull: Callable[[Transfer], None]
    verify: Callable[[Settings], bool]
    unavailable_hint: Callable[[Settings], str] = _default_hint


_REGISTRY: dict[str, BackupTarget] = {}


def register(target: BackupTarget) -> None:
    """Add (or replace) a target in the registry."""
    _REGISTRY[target.name] = target


def get_target(name: str) -> BackupTarget | None:
    """Look up a target by name; ``None`` if the name is unknown."""
    return _REGISTRY.get(name)


def target_names() -> list[str]:
    """Registered target names, for help text and diagnostics."""
    return sorted(_REGISTRY)


# --- local: a directory / mounted drive (also the test target) ----------------------------------


def _local_available(settings: Settings) -> bool:
    from .local import local_available

    return local_available(settings)


def _local_push(transfer: Transfer) -> str:
    from .local import local_push

    return local_push(transfer)


def _local_pull(transfer: Transfer) -> None:
    from .local import local_pull

    return local_pull(transfer)


def _local_verify(settings: Settings) -> bool:
    from .local import local_verify

    return local_verify(settings)


def _local_hint(settings: Settings) -> str:
    return "set KB_BACKUP_LOCAL_DIR to a writable directory (or run the backup-setup wizard)"


# --- rclone: any configured rclone remote (every cloud provider) ---------------------------------


def _rclone_available(settings: Settings) -> bool:
    from .rclone import rclone_target_available

    return rclone_target_available(settings)


def _rclone_push(transfer: Transfer) -> str:
    from .rclone import rclone_push

    return rclone_push(transfer)


def _rclone_pull(transfer: Transfer) -> None:
    from .rclone import rclone_pull

    return rclone_pull(transfer)


def _rclone_verify(settings: Settings) -> bool:
    from .rclone import rclone_verify

    return rclone_verify(settings)


def _rclone_hint(settings: Settings) -> str:
    from .rclone import rclone_available

    if not rclone_available(settings):
        return "install rclone and configure a remote (e.g. `rclone config`), then set KB_BACKUP_RCLONE_REMOTE"
    return "set KB_BACKUP_RCLONE_REMOTE to a configured rclone remote (e.g. gdrive:kb-backup)"


# --- none: "don't backup" (a valid, default target for the index) --------------------------------


def _none_push(transfer: Transfer) -> str:
    return "none"


def _none_pull(transfer: Transfer) -> None:
    return None


register(
    BackupTarget("local", _local_available, _local_push, _local_pull, _local_verify, _local_hint)
)
register(
    BackupTarget("rclone", _rclone_available, _rclone_push, _rclone_pull, _rclone_verify, _rclone_hint)
)
register(
    BackupTarget("none", lambda s: True, _none_push, _none_pull, lambda s: True)
)


def stage_files(base_dir: Path, files: list[Path], staging_dir: Path) -> None:
    """Copy each asset file into ``staging_dir`` preserving its path relative to ``base_dir``.

    Staging decouples *which* files make up an asset from *how* a target moves a tree, and guarantees
    only the intended files travel (e.g. the index's ``kb.sqlite`` without its ``-wal``/``-shm``).
    """
    import shutil

    for src in files:
        rel = src.resolve().relative_to(base_dir.resolve())
        dest = staging_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
