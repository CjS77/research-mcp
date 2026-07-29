"""The ``local`` backup target: copy an asset to a local directory or mounted drive.

This is the 3-2-1 "on-site copy" and the engine's test target — its unencrypted path is pure
Python (``shutil``), so the full push -> pull -> verify round-trip runs with no external tooling.
When encryption is enabled (the default), the copy is delegated to rclone's ``crypt`` overlay just
like the cloud target, so a local backup is encrypted at rest too.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..config import Settings
from .crypt import BackupError
from .targets import Transfer


def local_available(settings: Settings) -> bool:
    """True once a local backup directory is configured (its writability is checked by ``verify``)."""
    return settings.backup_local_dir is not None


def _asset_dir(settings: Settings, asset: str) -> Path:
    if settings.backup_local_dir is None:
        raise BackupError("no local backup directory configured (set KB_BACKUP_LOCAL_DIR)")
    return settings.backup_local_dir / asset


def _copy_tree(src: Path, dst: Path) -> None:
    """Copy every file under ``src`` into ``dst``, preserving the relative tree (idempotent)."""
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def local_push(transfer: Transfer) -> str:
    """Copy the staged tree into ``<local_dir>/<asset>`` (plaintext) or via crypt when encrypted."""
    settings = transfer.settings
    dest = _asset_dir(settings, transfer.asset)
    if transfer.encrypted:
        from .rclone import push_tree

        dest.mkdir(parents=True, exist_ok=True)
        push_tree(transfer.staging_dir, str(dest), encrypted=True, passphrase=transfer.passphrase, settings=settings)
    else:
        _copy_tree(transfer.staging_dir, dest)
    return f"local:{dest}"


def local_pull(transfer: Transfer) -> None:
    """Copy ``<local_dir>/<asset>`` back into the staging dir (decrypting via crypt when encrypted)."""
    settings = transfer.settings
    src = _asset_dir(settings, transfer.asset)
    if transfer.encrypted:
        from .rclone import pull_tree

        pull_tree(transfer.staging_dir, str(src), encrypted=True, passphrase=transfer.passphrase, settings=settings)
    else:
        if not src.exists():
            raise BackupError(f"no local backup found at {src}")
        _copy_tree(src, transfer.staging_dir)


def local_verify(settings: Settings) -> bool:
    """Reachable when the configured directory exists or its parent does (so it can be created)."""
    d = settings.backup_local_dir
    return d is not None and (d.exists() or d.parent.exists())
