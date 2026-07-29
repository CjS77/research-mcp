"""The backup manifest: per-file content hash + where each asset was pushed + when.

The manifest is the proof a backup can be restored. Every file that makes up an asset is recorded
with its SHA-256 (the same digest ``corpus`` uses for incremental indexing) and byte size; every
target the asset was pushed to is recorded with its location and timestamp. Restore fetches by the
manifest and *re-verifies every file's hash* after decryption — a backup that can't be proven to
restore is not a backup, exactly as ``acquire`` refuses to trust an unverified download.

The manifest itself holds only hashes and locations — never the passphrase, never file contents —
so it is safe to commit alongside the repo.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

MANIFEST_VERSION = 1


class FileEntry(BaseModel):
    """One backed-up file: its path relative to the asset's base dir, plus its digest and size."""

    path: str  # posix, relative to the asset base_dir — the stable identity across machines
    sha256: str
    size: int


class TargetRecord(BaseModel):
    """Where (and how) an asset was last pushed to one target."""

    location: str  # human-readable destination, e.g. "local:/mnt/backup/artifacts"
    encrypted: bool
    pushed_at: str  # ISO-8601 UTC


class AssetRecord(BaseModel):
    """The manifest entry for one asset: the files it comprises and the targets it reached."""

    files: list[FileEntry] = Field(default_factory=list)
    targets: dict[str, TargetRecord] = Field(default_factory=dict)


class BackupManifest(BaseModel):
    """The whole manifest: a version tag and a record per asset."""

    version: int = MANIFEST_VERSION
    assets: dict[str, AssetRecord] = Field(default_factory=dict)

    def asset(self, name: str) -> AssetRecord:
        """The record for ``name``, creating an empty one if this is its first push."""
        return self.assets.setdefault(name, AssetRecord())


def hash_file(path: Path) -> tuple[str, int]:
    """Return ``(sha256_hex, size_bytes)`` for a file, streamed so large indexes don't load into RAM."""
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def file_entries(base_dir: Path, files: list[Path]) -> list[FileEntry]:
    """Hash each file and record it relative to ``base_dir`` (sorted for a stable manifest)."""
    entries = []
    for path in sorted(files):
        digest, size = hash_file(path)
        rel = path.resolve().relative_to(base_dir.resolve()).as_posix()
        entries.append(FileEntry(path=rel, sha256=digest, size=size))
    return entries


def verify_files(base_dir: Path, entries: list[FileEntry]) -> list[str]:
    """Return the rel-paths whose on-disk bytes do **not** match the manifest (missing or corrupted)."""
    bad = []
    for entry in entries:
        path = base_dir / entry.path
        if not path.exists():
            bad.append(entry.path)
            continue
        digest, size = hash_file(path)
        if digest != entry.sha256 or size != entry.size:
            bad.append(entry.path)
    return bad


@dataclass
class DriftReport:
    """How the live asset differs from what the manifest last recorded as pushed."""

    added: list[str]  # present on disk, not in the manifest
    removed: list[str]  # in the manifest, gone from disk
    changed: list[str]  # present in both, different hash

    @property
    def clean(self) -> bool:
        return not (self.added or self.removed or self.changed)

    def as_dict(self) -> dict[str, list[str]]:
        return {"added": self.added, "removed": self.removed, "changed": self.changed}


def drift(base_dir: Path, current: list[Path], recorded: list[FileEntry]) -> DriftReport:
    """Compare the live files under ``base_dir`` against the manifest's recorded entries."""
    recorded_by_path = {e.path: e for e in recorded}
    current_by_path = {
        p.resolve().relative_to(base_dir.resolve()).as_posix(): p for p in current
    }
    added = sorted(set(current_by_path) - set(recorded_by_path))
    removed = sorted(set(recorded_by_path) - set(current_by_path))
    changed = []
    for rel in sorted(set(current_by_path) & set(recorded_by_path)):
        digest, size = hash_file(current_by_path[rel])
        entry = recorded_by_path[rel]
        if digest != entry.sha256 or size != entry.size:
            changed.append(rel)
    return DriftReport(added=added, removed=removed, changed=changed)


def load_manifest(path: Path) -> BackupManifest:
    """Load the manifest, or an empty one if none has been written yet."""
    if not path.exists():
        return BackupManifest()
    return BackupManifest.model_validate_json(path.read_text(encoding="utf-8"))


def save_manifest(manifest: BackupManifest, path: Path) -> None:
    """Persist the manifest as pretty JSON (safe to commit — it holds no secrets)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
