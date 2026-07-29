"""Durable, encrypted backup of the KB's two irreplaceable assets, plus a one-command rebuild.

The token-expensive **distillation artifacts** (``work/distilled/<stem>/*.md``) and the **index**
(``work/data/kb.sqlite``) are protected via a target registry (:mod:`.targets`) — ``rclone`` (any
cloud provider), ``local`` (a directory / mounted drive), and ``none`` — with client-side encryption
through rclone's ``crypt`` overlay (:mod:`.rclone`, :mod:`.crypt`) on by default. Every push records a
manifest (:mod:`.manifest`) of per-file hashes + locations and re-verifies the round-trip; restore
verifies each hash before placing files back. :func:`rebuild` reconstructs the index from the corpus
+ committed artifacts so the index need not be backed up at all.

Public surface (used by the CLI and the refresh cron):
"""

from __future__ import annotations

from .crypt import BackupError
from .engine import (
    ASSETS,
    AssetStatus,
    PushResult,
    RestoreResult,
    StatusReport,
    push,
    rebuild,
    restore,
    status,
)
from .targets import BackupTarget, get_target, register, target_names

__all__ = [
    "ASSETS",
    "AssetStatus",
    "BackupError",
    "BackupTarget",
    "PushResult",
    "RestoreResult",
    "StatusReport",
    "get_target",
    "push",
    "rebuild",
    "register",
    "restore",
    "status",
    "target_names",
]
