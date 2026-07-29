"""Backup orchestration: resolve assets to files, push/restore/status, and the one-command rebuild.

Two assets, split by value with different defaults:

- **artifacts** — the token-expensive distillation output under ``work/distilled/``; default target a
  cloud provider (rclone), because re-distilling costs hours of LLM transcription.
- **index** — ``work/data/kb.sqlite``; default target ``none`` ("don't backup"), because
  :func:`rebuild` reconstructs it from ``reference/`` + the committed artifacts in one command.

Every push writes the manifest (per-file hash + target location + timestamp) and, unless disabled,
immediately pulls the asset back into a temp dir and re-verifies every hash — a backup that can't be
proven to restore is not a backup. Restore is symmetric: download, decrypt, **verify against the
manifest**, then place into the working tree. Nothing here is called from ``index`` or ``distill``;
backup runs only on an explicit ``backup push`` (or the refresh cron), keeping serving offline.
"""

from __future__ import annotations

import sqlite3
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import Settings, get_settings
from .crypt import BackupError, get_passphrase, is_encrypted
from .manifest import (
    AssetRecord,
    DriftReport,
    FileEntry,
    TargetRecord,
    drift,
    file_entries,
    load_manifest,
    save_manifest,
    verify_files,
)
from .rclone import rclone_available
from .targets import Transfer, get_target, stage_files

if TYPE_CHECKING:
    from ..index import IndexSummary

ASSETS: tuple[str, ...] = ("artifacts", "index")


def _now() -> str:
    return datetime.now(UTC).isoformat()


# --- Asset resolution ----------------------------------------------------------------------------


@dataclass
class AssetSpec:
    """An asset resolved to concrete files under a base dir (rel paths are ``file.relative_to(base)``)."""

    name: str
    base_dir: Path
    files: list[Path]


def targets_for_asset(settings: Settings, asset: str) -> tuple[str, ...]:
    """The configured target-name list for an asset (artifacts -> cloud, index -> none, by default)."""
    if asset == "artifacts":
        return settings.backup_artifacts_targets
    if asset == "index":
        return settings.backup_index_targets
    raise BackupError(f"unknown asset {asset!r} (expected one of {ASSETS})")


def asset_base_dir(settings: Settings, asset: str) -> Path:
    """Where an asset's files live and are restored to."""
    if asset == "artifacts":
        return settings.distilled_dir
    if asset == "index":
        return settings.db_path.parent
    raise BackupError(f"unknown asset {asset!r} (expected one of {ASSETS})")


def _checkpoint_index(db_path: Path) -> None:
    """Best-effort WAL checkpoint so the single ``kb.sqlite`` file is a complete point-in-time copy."""
    try:
        con = sqlite3.connect(db_path)
        try:
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            con.close()
    except sqlite3.Error:
        pass  # DB busy/locked — the residual -wal is included below so the backup stays complete


def asset_files(settings: Settings, asset: str, *, checkpoint: bool = True) -> list[Path]:
    """The files that make up an asset (empty when there is nothing to back up)."""
    if asset == "artifacts":
        base = settings.distilled_dir
        return sorted(p for p in base.rglob("*") if p.is_file()) if base.exists() else []
    if asset == "index":
        if not settings.db_path.exists():
            return []
        if checkpoint:
            _checkpoint_index(settings.db_path)
        files = [settings.db_path]
        wal = Path(f"{settings.db_path}-wal")
        if wal.exists() and wal.stat().st_size > 0:  # checkpoint could not truncate — keep it consistent
            files.append(wal)
        return files
    raise BackupError(f"unknown asset {asset!r} (expected one of {ASSETS})")


def asset_spec(settings: Settings, asset: str, *, checkpoint: bool = True) -> AssetSpec:
    return AssetSpec(asset, asset_base_dir(settings, asset), asset_files(settings, asset, checkpoint=checkpoint))


# --- Push ----------------------------------------------------------------------------------------


@dataclass
class PushResult:
    asset: str
    target: str
    location: str | None
    n_files: int
    status: str  # "pushed" | "noop" | "unavailable" | "error"
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {
            "asset": self.asset, "target": self.target, "location": self.location,
            "n_files": self.n_files, "status": self.status, "detail": self.detail,
        }


def _resolve_passphrase(settings: Settings, target_name: str) -> tuple[bool, str | None, str | None]:
    """(encrypted, passphrase, blocking_reason). A reason is set when encryption cannot proceed."""
    if not is_encrypted(settings, target_name):
        return False, None, None
    if not rclone_available(settings):
        return True, None, "encryption is on but rclone (the crypt engine) is not installed"
    passphrase = get_passphrase(settings)
    if not passphrase:
        return True, None, "encryption is on but no crypt passphrase (keychain / KB_BACKUP_PASSPHRASE / prompt)"
    return True, passphrase, None


def _push_one(
    settings: Settings, asset: str, target_name: str, spec: AssetSpec, entries: list[FileEntry]
) -> tuple[PushResult, TargetRecord | None]:
    target = get_target(target_name)
    if target is None:
        return PushResult(asset, target_name, None, 0, "error", f"unknown target '{target_name}'"), None
    if not target.available(settings):
        return PushResult(asset, target_name, None, 0, "unavailable", target.unavailable_hint(settings)), None

    encrypted, passphrase, blocked = _resolve_passphrase(settings, target_name)
    if blocked is not None:
        return PushResult(asset, target_name, None, 0, "unavailable", f"{blocked} — run the backup-setup wizard"), None

    try:
        with tempfile.TemporaryDirectory() as td:
            staging = Path(td) / "src"
            staging.mkdir()
            stage_files(spec.base_dir, spec.files, staging)
            location = target.push(Transfer(asset, staging, encrypted, passphrase, settings))
        if settings.backup_verify_pushes:
            _verify_push(settings, asset, target_name, entries, encrypted, passphrase)
    except BackupError as exc:
        return PushResult(asset, target_name, None, 0, "error", str(exc)), None

    record = TargetRecord(location=location, encrypted=encrypted, pushed_at=_now())
    detail = f"{len(entries)} file(s){' encrypted' if encrypted else ''} -> {location}"
    return PushResult(asset, target_name, location, len(entries), "pushed", detail), record


def _verify_push(
    settings: Settings, asset: str, target_name: str, entries: list[FileEntry], encrypted: bool, passphrase: str | None
) -> None:
    """Pull the just-pushed asset back into a temp dir and re-verify every hash against the manifest."""
    target = get_target(target_name)
    assert target is not None  # availability already checked in _push_one
    with tempfile.TemporaryDirectory() as td:
        check = Path(td)
        target.pull(Transfer(asset, check, encrypted, passphrase, settings))
        bad = verify_files(check, entries)
    if bad:
        raise BackupError(f"upload verification failed for {len(bad)} file(s): {', '.join(bad[:5])}")


def push(
    settings: Settings | None = None,
    assets: Iterable[str] = ASSETS,
    on_result: Callable[[PushResult], None] | None = None,
) -> list[PushResult]:
    """Encrypt + upload each asset to its configured target(s), verifying each upload; write the manifest."""
    settings = settings or get_settings()
    manifest = load_manifest(settings.backup_manifest_path)
    results: list[PushResult] = []

    def emit(r: PushResult) -> PushResult:
        if on_result is not None:
            on_result(r)
        results.append(r)
        return r

    dirty = False
    for asset in assets:
        active = [t for t in targets_for_asset(settings, asset) if t != "none"]
        if not active:
            emit(PushResult(asset, "none", None, 0, "noop",
                            'no backup target configured ("none") — run the backup-setup wizard'))
            continue
        spec = asset_spec(settings, asset)
        if not spec.files:
            emit(PushResult(asset, ",".join(active), None, 0, "noop", f"nothing to back up for '{asset}'"))
            continue
        entries = file_entries(spec.base_dir, spec.files)
        for target_name in active:
            result, record = _push_one(settings, asset, target_name, spec, entries)
            if record is not None:
                arec = manifest.asset(asset)
                arec.files = entries
                arec.targets[target_name] = record
                dirty = True
            emit(result)

    if dirty:
        save_manifest(manifest, settings.backup_manifest_path)
    return results


# --- Restore -------------------------------------------------------------------------------------


@dataclass
class RestoreResult:
    asset: str
    target: str
    n_files: int
    verified: int
    status: str  # "restored" | "noop" | "unavailable" | "error"
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {
            "asset": self.asset, "target": self.target, "n_files": self.n_files,
            "verified": self.verified, "status": self.status, "detail": self.detail,
        }


def _place_files(staging: Path, entries: list[FileEntry], base_dir: Path) -> None:
    """Copy verified files from the staging dir into the working tree at their recorded rel paths."""
    import shutil

    for entry in entries:
        dst = base_dir / entry.path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(staging / entry.path, dst)


def _choose_restore_target(settings: Settings, recorded: dict[str, TargetRecord], prefer: str | None) -> str | None:
    """The target to restore from: an explicit choice if recorded + available, else the first that is."""
    order = [prefer] if prefer else list(recorded)
    for name in order:
        if name not in recorded:
            continue
        target = get_target(name)
        if target is not None and target.available(settings):
            return name
    return None


def _restore_asset(
    settings: Settings, asset: str, manifest_assets: dict[str, AssetRecord], prefer: str | None
) -> RestoreResult:
    arec = manifest_assets.get(asset)
    if arec is None or not arec.targets:
        return RestoreResult(asset, "-", 0, 0, "noop",
                             "nothing recorded in the manifest (never pushed?) — run the backup-setup wizard")

    chosen = _choose_restore_target(settings, arec.targets, prefer)
    if chosen is None:
        named = prefer or ", ".join(arec.targets)
        return RestoreResult(asset, named, 0, 0, "unavailable",
                             f"no available recorded target to restore from ({named}) — re-auth the remote / set the dir")

    trec = arec.targets[chosen]
    encrypted = trec.encrypted
    passphrase = get_passphrase(settings) if encrypted else None
    if encrypted and not passphrase:
        return RestoreResult(asset, chosen, 0, 0, "unavailable",
                             "backup is encrypted but no passphrase available (keychain / KB_BACKUP_PASSPHRASE)")

    target = get_target(chosen)
    assert target is not None  # chosen came from _choose_restore_target, which checked availability
    base = asset_base_dir(settings, asset)
    try:
        with tempfile.TemporaryDirectory() as td:
            staging = Path(td)
            target.pull(Transfer(asset, staging, encrypted, passphrase, settings))
            bad = verify_files(staging, arec.files)
            if bad:
                raise BackupError(f"hash verification failed for {len(bad)} file(s): {', '.join(bad[:5])}")
            _place_files(staging, arec.files, base)
    except BackupError as exc:
        return RestoreResult(asset, chosen, 0, 0, "error", str(exc))

    n = len(arec.files)
    return RestoreResult(asset, chosen, n, n, "restored", f"restored {n} file(s) from {trec.location}")


def restore(
    settings: Settings | None = None,
    assets: Iterable[str] = ASSETS,
    target: str | None = None,
    on_result: Callable[[RestoreResult], None] | None = None,
) -> list[RestoreResult]:
    """Download + decrypt + verify each asset from the manifest, placing it back into the working tree."""
    settings = settings or get_settings()
    manifest = load_manifest(settings.backup_manifest_path)
    results: list[RestoreResult] = []
    for asset in assets:
        result = _restore_asset(settings, asset, manifest.assets, target)
        if on_result is not None:
            on_result(result)
        results.append(result)
    return results


# --- Status --------------------------------------------------------------------------------------


@dataclass
class AssetStatus:
    asset: str
    targets: list[str]
    n_files: int
    target_details: list[dict[str, object]]
    drift: DriftReport

    def as_dict(self) -> dict[str, object]:
        return {
            "asset": self.asset, "targets": list(self.targets), "n_files": self.n_files,
            "target_details": self.target_details, "drift": self.drift.as_dict(),
            "drift_clean": self.drift.clean,
        }


@dataclass
class StatusReport:
    manifest_path: str
    encryption_default: bool
    assets: list[AssetStatus] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "manifest_path": self.manifest_path, "encryption_default": self.encryption_default,
            "assets": [a.as_dict() for a in self.assets],
        }


def status(settings: Settings | None = None) -> StatusReport:
    """Configured targets, availability + last-push time per asset/target, and drift since last push."""
    settings = settings or get_settings()
    manifest = load_manifest(settings.backup_manifest_path)
    report = StatusReport(str(settings.backup_manifest_path), settings.backup_encrypt)

    for asset in ASSETS:
        resolved = targets_for_asset(settings, asset)
        files = asset_files(settings, asset, checkpoint=False)  # read-only: never checkpoint in status
        base = asset_base_dir(settings, asset)
        arec = manifest.assets.get(asset)

        details: list[dict[str, object]] = []
        for name in resolved:
            target = get_target(name)
            rec = arec.targets.get(name) if arec else None
            details.append({
                "name": name,
                "available": bool(target and target.available(settings)),
                "encrypted": is_encrypted(settings, name) if name != "none" else False,
                "last_push": rec.pushed_at if rec else None,
                "location": rec.location if rec else None,
            })

        recorded = arec.files if arec else []
        report.assets.append(AssetStatus(asset, list(resolved), len(files), details, drift(base, files, recorded)))
    return report


# --- Rebuild -------------------------------------------------------------------------------------


def rebuild(settings: Settings | None = None) -> IndexSummary:
    """Reconstruct ``work/data/kb.sqlite`` from ``reference/`` + committed distillation artifacts.

    This is the index's safety net — the reason the index defaults to "don't backup". It deletes any
    existing DB and re-runs the normal pipeline, which reuses the committed ``distilled/<stem>/llm.md``
    transcriptions (never re-distilling: the heavy ``claude_cli`` backend is not index-time-safe), so a
    fresh clone with the artifacts present becomes queryable in one command.
    """
    from ..index import index_corpus

    settings = settings or get_settings()
    for suffix in ("", "-wal", "-shm"):
        Path(f"{settings.db_path}{suffix}").unlink(missing_ok=True)
    return index_corpus(settings, roots=[settings.reference_dir])
