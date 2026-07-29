"""Thin wrapper over the ``rclone`` binary — the one external tool the backup engine shells out to.

rclone is not vendored: it is invoked as a subprocess so a single adapter reaches *every* provider
it supports (Google Drive, Dropbox, OneDrive, Proton, S3, …) and gives chunked, resumable,
checksummed transfer for free. It is also the encryption engine: client-side encryption is rclone's
``crypt`` overlay, applied on the fly via a connection string so no plaintext key is ever written to
an rclone config file.

Every subprocess call funnels through :func:`run_rclone` — the single seam tests monkeypatch, so no
test needs a real ``rclone`` on PATH (mirroring how ``acquire`` fakes its httpx client).
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

from ..config import Settings
from .crypt import BackupError

if TYPE_CHECKING:
    from pathlib import Path

    from .targets import Transfer


def rclone_available(settings: Settings) -> bool:
    """True when the ``rclone`` binary is on PATH (any rclone-backed transfer can run)."""
    return shutil.which(settings.backup_rclone_bin) is not None


def run_rclone(args: list[str], settings: Settings) -> subprocess.CompletedProcess[str]:
    """Run ``rclone <args>`` and capture output. The single subprocess seam for the whole engine."""
    cmd = [settings.backup_rclone_bin, *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=settings.backup_rclone_timeout_s)


def obscure(passphrase: str, settings: Settings) -> str:
    """Obscure a passphrase with ``rclone obscure`` — the form a crypt connection string requires.

    (``obscure`` is reversible and *not* a security boundary; it only keeps the passphrase out of a
    plaintext argv/config. The security comes from the crypt overlay itself.)
    """
    proc = run_rclone(["obscure", passphrase], settings)
    if proc.returncode != 0 or not proc.stdout.strip():
        raise BackupError(f"rclone obscure failed: {proc.stderr.strip() or 'no output'}")
    return proc.stdout.strip()


def crypt_connection(base_remote: str, obscured_pw: str) -> str:
    """An on-the-fly ``crypt`` connection string wrapping ``base_remote`` (a remote:path or local path).

    Config-less, so the passphrase never lands in an rclone config file. Filenames and contents are
    both encrypted at rest; rclone decrypts them transparently on the reverse copy.
    """
    return f":crypt,remote='{base_remote}',password='{obscured_pw}':"


def _endpoint(base_remote: str, *, encrypted: bool, passphrase: str | None, settings: Settings) -> str:
    """The rclone endpoint for ``base_remote`` — wrapped in a crypt overlay when encryption is on."""
    if not encrypted:
        return base_remote
    if not passphrase:
        raise BackupError(
            "encryption is on but no crypt passphrase is available "
            "(keychain / KB_BACKUP_PASSPHRASE / prompt) — run the backup-setup wizard"
        )
    return crypt_connection(base_remote, obscure(passphrase, settings))


def push_tree(
    staging_dir: Path, base_remote: str, *, encrypted: bool, passphrase: str | None, settings: Settings
) -> None:
    """Copy the staged tree up to ``base_remote`` (encrypting on the fly when enabled)."""
    dest = _endpoint(base_remote, encrypted=encrypted, passphrase=passphrase, settings=settings)
    proc = run_rclone(["copy", str(staging_dir), dest], settings)
    if proc.returncode != 0:
        raise BackupError(f"rclone push failed: {proc.stderr.strip() or proc.stdout.strip() or 'nonzero exit'}")


def pull_tree(
    staging_dir: Path, base_remote: str, *, encrypted: bool, passphrase: str | None, settings: Settings
) -> None:
    """Copy the tree at ``base_remote`` back down into ``staging_dir`` (decrypting on the fly)."""
    src = _endpoint(base_remote, encrypted=encrypted, passphrase=passphrase, settings=settings)
    proc = run_rclone(["copy", src, str(staging_dir)], settings)
    if proc.returncode != 0:
        raise BackupError(f"rclone pull failed: {proc.stderr.strip() or proc.stdout.strip() or 'nonzero exit'}")


def reachable(base_remote: str, settings: Settings) -> bool:
    """Best-effort reachability check for a plain (unencrypted) remote path, for `verify` / `status`.

    ``rclone lsf`` listing the destination proves the remote authenticates; a not-yet-created backup
    directory ("directory not found") still counts as reachable — the remote works, it is just empty.
    """
    if not rclone_available(settings):
        return False
    proc = run_rclone(["lsf", "--max-depth", "1", base_remote], settings)
    if proc.returncode == 0:
        return True
    return "not found" in proc.stderr.lower() or "doesn't exist" in proc.stderr.lower()


# --- The `rclone` backup target (registered in targets.py) ---------------------------------------


def _rclone_base(settings: Settings, asset: str) -> str:
    """The remote:path an asset lives under, e.g. ``gdrive:kb-backup/artifacts``."""
    return f"{settings.backup_rclone_remote.rstrip('/')}/{asset}"


def rclone_target_available(settings: Settings) -> bool:
    """Usable when the binary is on PATH *and* a remote is named (auth is set up out-of-band)."""
    return rclone_available(settings) and bool(settings.backup_rclone_remote)


def rclone_push(transfer: Transfer) -> str:
    """Upload the staged tree to ``<remote>/<asset>`` (encrypting on the fly when enabled)."""
    settings = transfer.settings
    base = _rclone_base(settings, transfer.asset)
    push_tree(transfer.staging_dir, base, encrypted=transfer.encrypted, passphrase=transfer.passphrase, settings=settings)
    return f"rclone:{base}"


def rclone_pull(transfer: Transfer) -> None:
    """Download ``<remote>/<asset>`` into the staging dir (decrypting on the fly when enabled)."""
    settings = transfer.settings
    base = _rclone_base(settings, transfer.asset)
    pull_tree(transfer.staging_dir, base, encrypted=transfer.encrypted, passphrase=transfer.passphrase, settings=settings)


def rclone_verify(settings: Settings) -> bool:
    """Best-effort reachability of the configured remote (auth works, dir may not exist yet)."""
    if not rclone_target_available(settings):
        return False
    return reachable(settings.backup_rclone_remote.rstrip("/"), settings)
