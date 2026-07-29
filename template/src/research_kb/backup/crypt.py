"""Encryption policy and the crypt passphrase — sourced from the OS keychain, never the repo.

Backups are encrypted client-side with rclone's ``crypt`` overlay (see :mod:`.rclone`), on by
default for every target. This module answers two questions the engine asks per target:

- *Is this target encrypted?* — ``is_encrypted`` (global default on; opt out per target).
- *What is the passphrase?* — ``get_passphrase``, resolved from the **OS keychain** first, then the
  ``KB_BACKUP_PASSPHRASE`` env var, then an interactive prompt. The plaintext passphrase is **never**
  written to the repo or the cloud; generating it and caching it in the keychain is the wizard's job,
  this module only *reads* it.

Keychain access goes through the tiny ``_keyring_get`` seam so tests monkeypatch it and never touch
the real OS keychain.
"""

from __future__ import annotations

import os
import sys

from ..config import Settings

PASSPHRASE_ENV = "KB_BACKUP_PASSPHRASE"


class BackupError(RuntimeError):
    """A backup operation could not proceed (misconfiguration, missing key, transfer failure)."""


def is_encrypted(settings: Settings, target_name: str) -> bool:
    """Whether pushes to ``target_name`` are encrypted: on by default, opt out per target."""
    return settings.backup_encrypt and target_name not in settings.backup_plaintext_targets


def _keyring_get(service: str, account: str) -> str | None:
    """Read a secret from the OS keychain via the ``keyring`` library. The single seam tests patch.

    Returns ``None`` (never raises) when ``keyring`` is missing or has no configured backend, so a
    headless box without a keychain falls through cleanly to the env-var / prompt fallbacks.
    """
    try:
        import keyring
    except Exception:  # noqa: BLE001 — keyring absent or import-time backend failure
        return None
    try:
        return keyring.get_password(service, account)
    except Exception:  # noqa: BLE001 — no usable keychain backend on this host
        return None


def get_passphrase(settings: Settings, *, allow_prompt: bool = True) -> str | None:
    """Resolve the crypt passphrase: OS keychain -> ``KB_BACKUP_PASSPHRASE`` -> interactive prompt.

    Returns ``None`` when no passphrase can be obtained non-interactively and no TTY is available to
    prompt — the caller then reports the target as unavailable and points at the setup wizard rather
    than silently pushing plaintext.
    """
    keychain = _keyring_get(settings.backup_keyring_service, settings.backup_keyring_account)
    if keychain:
        return keychain
    env = os.environ.get(PASSPHRASE_ENV)
    if env:
        return env
    if allow_prompt and sys.stdin.isatty():
        import click

        entered = click.prompt("backup crypt passphrase", hide_input=True, default="", show_default=False)
        return entered or None
    return None
