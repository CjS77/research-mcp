"""Backup engine: registry, manifest, local round-trip, encryption seam, defaults, rebuild, invariant.

No test touches a real ``rclone`` binary or the real OS keychain. The pure/local paths (local target
round-trip, manifest hashing + verification, registry, none/no-op, per-asset defaults, rebuild) run
unconditionally; the crypt path is exercised with a fake ``rclone`` (asserting the crypt overlay is
invoked and the round-trip verifies), and a real end-to-end crypt round-trip is gated on rclone being
installed. Keychain access is always monkeypatched via the ``crypt._keyring_get`` seam.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_kb.backup import (
    crypt,
    engine,
    manifest,
    push,
    rclone,
    rebuild,
    restore,
    status,
    target_names,
    targets,
)
from research_kb.config import Settings
from research_kb.db import connect
from research_kb.index import index_corpus
from research_kb.service import search_service

# --- fixtures / helpers --------------------------------------------------------------------------


def _bsettings(tmp_path: Path, **over) -> Settings:
    ref = tmp_path / "ref"
    ref.mkdir(exist_ok=True)
    kw: dict = dict(
        db_path=tmp_path / "work" / "data" / "kb.sqlite",
        distilled_dir=tmp_path / "work" / "distilled",
        base_dir=tmp_path,
        reference_dir=ref,
        docs_dir=tmp_path / "docs",
        embed_backend="hashing",
        embed_dim=256,
        backup_local_dir=tmp_path / "backupdir",
        backup_artifacts_targets=("local",),
        backup_index_targets=("none",),
        backup_encrypt=False,  # plaintext local: the fully-tested, rclone-free path
    )
    kw.update(over)
    return Settings(**kw)


def _make_artifacts(s: Settings) -> Path:
    d = s.distilled_dir / "paper1"
    d.mkdir(parents=True, exist_ok=True)
    (d / "verbatim.md").write_text("VERBATIM one", encoding="utf-8")
    (d / "llm.md").write_text("LLM transcription two", encoding="utf-8")
    (d / "enriched.md").write_text("ENRICHED three", encoding="utf-8")
    return d


class _FakeRclone:
    """A stand-in for the ``rclone`` binary that simulates ``obscure`` + ``copy`` on the local FS.

    ``crypt`` is modelled as a *passthrough* copy (no real encryption) so the round-trip + hash
    verification still exercise the encrypted code path end to end without a real rclone.
    """

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str], settings: Settings):  # noqa: ANN204 — mimics run_rclone
        self.calls.append(list(args))
        if args[0] == "obscure":
            return SimpleNamespace(returncode=0, stdout=f"OBSCURED::{args[1]}", stderr="")
        if args[0] == "copy":
            src = self._real(args[1])
            dst = self._real(args[2])
            Path(dst).mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst, dirs_exist_ok=True)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    @staticmethod
    def _real(arg: str) -> str:
        m = re.match(r":crypt,remote='([^']*)',", arg)
        return m.group(1) if m else arg


# --- registry ------------------------------------------------------------------------------------


def test_shipped_targets_registered():
    assert {"local", "rclone", "none"} <= set(target_names())
    assert targets.get_target("bogus") is None


def test_none_target_is_always_available():
    t = targets.get_target("none")
    assert t is not None and t.available(Settings()) is True


# --- manifest: hashing + verification + drift ----------------------------------------------------


def test_hash_file_and_entries(tmp_path: Path):
    base = tmp_path / "base"
    (base / "sub").mkdir(parents=True)
    f = base / "sub" / "a.md"
    f.write_text("hello", encoding="utf-8")
    digest, size = manifest.hash_file(f)
    assert size == 5 and len(digest) == 64
    entries = manifest.file_entries(base, [f])
    assert entries[0].path == "sub/a.md" and entries[0].sha256 == digest


def test_verify_files_catches_corruption(tmp_path: Path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "x.md").write_text("original", encoding="utf-8")
    entries = manifest.file_entries(base, [base / "x.md"])
    assert manifest.verify_files(base, entries) == []
    (base / "x.md").write_text("tampered!", encoding="utf-8")  # same-length change still caught by hash
    assert manifest.verify_files(base, entries) == ["x.md"]
    (base / "x.md").unlink()
    assert manifest.verify_files(base, entries) == ["x.md"]  # missing also flagged


def test_drift_detects_added_removed_changed(tmp_path: Path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "keep.md").write_text("keep", encoding="utf-8")
    (base / "gone.md").write_text("gone", encoding="utf-8")
    recorded = manifest.file_entries(base, [base / "keep.md", base / "gone.md"])
    (base / "gone.md").unlink()
    (base / "keep.md").write_text("changed", encoding="utf-8")
    (base / "new.md").write_text("new", encoding="utf-8")
    current = [base / "keep.md", base / "new.md"]
    d = manifest.drift(base, current, recorded)
    assert d.added == ["new.md"] and d.removed == ["gone.md"] and d.changed == ["keep.md"]
    assert not d.clean


def test_manifest_save_load_roundtrip(tmp_path: Path):
    m = manifest.BackupManifest()
    arec = m.asset("artifacts")
    arec.files = [manifest.FileEntry(path="p/llm.md", sha256="a" * 64, size=3)]
    arec.targets["local"] = manifest.TargetRecord(location="local:/b", encrypted=True, pushed_at="2026-01-01T00:00:00Z")
    path = tmp_path / "backup-manifest.json"
    manifest.save_manifest(m, path)
    again = manifest.load_manifest(path)
    assert again.assets["artifacts"].files[0].path == "p/llm.md"
    assert again.assets["artifacts"].targets["local"].encrypted is True
    assert manifest.load_manifest(tmp_path / "missing.json").assets == {}  # absent -> empty


# --- per-asset target resolution + defaults ------------------------------------------------------


def test_default_targets_artifacts_cloud_index_none():
    s = Settings()
    assert engine.targets_for_asset(s, "artifacts") == ("rclone",)
    assert engine.targets_for_asset(s, "index") == ("none",)


def test_asset_files_resolution(tmp_path: Path):
    s = _bsettings(tmp_path)
    _make_artifacts(s)
    files = engine.asset_files(s, "artifacts")
    assert {p.name for p in files} == {"verbatim.md", "llm.md", "enriched.md"}
    assert engine.asset_files(s, "index") == []  # no DB yet


# --- local plaintext round-trip: push -> pull -> verify ------------------------------------------


def test_local_plaintext_push_restore_roundtrip(tmp_path: Path):
    s = _bsettings(tmp_path)
    art = _make_artifacts(s)

    results = push(s, ("artifacts",))
    assert [r.status for r in results] == ["pushed"]
    assert results[0].n_files == 3
    # files landed in the local backup dir under the asset name, structure preserved
    assert (s.backup_local_dir / "artifacts" / "paper1" / "llm.md").read_text() == "LLM transcription two"
    # manifest written with per-file hashes + the target location
    m = manifest.load_manifest(s.backup_manifest_path)
    assert len(m.assets["artifacts"].files) == 3
    assert m.assets["artifacts"].targets["local"].encrypted is False

    # wipe the working copy, then restore from the backup and re-verify hashes
    shutil.rmtree(art)
    assert not art.exists()
    rres = restore(s, ("artifacts",))
    assert [r.status for r in rres] == ["restored"]
    assert rres[0].verified == 3
    assert (art / "llm.md").read_text() == "LLM transcription two"


def test_restore_rejects_corrupted_backup(tmp_path: Path):
    s = _bsettings(tmp_path)
    _make_artifacts(s)
    push(s, ("artifacts",))
    # tamper with the backed-up copy: restore must catch the hash mismatch and refuse
    (s.backup_local_dir / "artifacts" / "paper1" / "llm.md").write_text("CORRUPTED", encoding="utf-8")
    rres = restore(s, ("artifacts",))
    assert rres[0].status == "error"
    assert "verification failed" in rres[0].detail


def test_push_verification_catches_bad_upload(tmp_path: Path, monkeypatch):
    # backup_verify_pushes (default) re-pulls each upload and re-hashes it against the manifest, so a
    # lossy/corrupted upload is caught at push time — the target is not recorded as good.
    from research_kb.backup import local as local_mod

    s = _bsettings(tmp_path)
    _make_artifacts(s)
    real_push = local_mod.local_push

    def lossy_push(transfer):
        loc = real_push(transfer)
        (s.backup_local_dir / "artifacts" / "paper1" / "llm.md").unlink()  # simulate a dropped file
        return loc

    monkeypatch.setattr(local_mod, "local_push", lossy_push)
    res = push(s, ("artifacts",))
    assert res[0].status == "error"
    assert "verification failed" in res[0].detail
    assert not s.backup_manifest_path.exists()  # a failed push records nothing


# --- none / "don't backup" no-op -----------------------------------------------------------------


def test_index_default_is_dont_backup_noop(tmp_path: Path):
    s = _bsettings(tmp_path)  # index target defaults to ("none",)
    res = push(s, ("index",))
    assert res[0].status == "noop"
    assert "wizard" in res[0].detail
    assert not s.backup_manifest_path.exists()  # nothing recorded


def test_restore_with_no_manifest_is_noop(tmp_path: Path):
    s = _bsettings(tmp_path)
    res = restore(s, ("artifacts",))
    assert res[0].status == "noop"
    assert "wizard" in res[0].detail


# --- unavailable targets point at the wizard -----------------------------------------------------


def test_rclone_target_unavailable_without_binary(tmp_path: Path, monkeypatch):
    # rclone remote configured but the binary is absent -> unavailable, hint points at install/wizard.
    monkeypatch.setattr(rclone.shutil, "which", lambda _bin: None)
    s = _bsettings(tmp_path, backup_artifacts_targets=("rclone",), backup_rclone_remote="gdrive:kb", backup_encrypt=False)
    _make_artifacts(s)
    res = push(s, ("artifacts",))
    assert res[0].status == "unavailable"
    assert "rclone" in res[0].detail.lower()


def test_encryption_requires_rclone(tmp_path: Path, monkeypatch):
    # Encryption on (default) but no rclone -> local push is unavailable with a clear message.
    monkeypatch.setattr(engine, "rclone_available", lambda _s: False)
    s = _bsettings(tmp_path, backup_encrypt=True)
    _make_artifacts(s)
    res = push(s, ("artifacts",))
    assert res[0].status == "unavailable"
    assert "rclone" in res[0].detail.lower() and "wizard" in res[0].detail


# --- passphrase resolution / keyring seam (never touches the real keychain) ----------------------


def test_get_passphrase_prefers_keychain(monkeypatch):
    monkeypatch.setattr(crypt, "_keyring_get", lambda service, account: "from-keychain")
    monkeypatch.setenv(crypt.PASSPHRASE_ENV, "from-env")
    assert crypt.get_passphrase(Settings()) == "from-keychain"


def test_get_passphrase_falls_back_to_env(monkeypatch):
    monkeypatch.setattr(crypt, "_keyring_get", lambda service, account: None)
    monkeypatch.setenv(crypt.PASSPHRASE_ENV, "from-env")
    assert crypt.get_passphrase(Settings()) == "from-env"


def test_get_passphrase_none_when_nothing_available(monkeypatch):
    monkeypatch.setattr(crypt, "_keyring_get", lambda service, account: None)
    monkeypatch.delenv(crypt.PASSPHRASE_ENV, raising=False)
    monkeypatch.setattr(crypt.sys.stdin, "isatty", lambda: False)  # no TTY -> no prompt
    assert crypt.get_passphrase(Settings()) is None


def test_is_encrypted_default_on_toggle_per_target():
    s = Settings(backup_encrypt=True, backup_plaintext_targets=("local",))
    assert crypt.is_encrypted(s, "rclone") is True
    assert crypt.is_encrypted(s, "local") is False
    assert crypt.is_encrypted(Settings(backup_encrypt=False), "rclone") is False


# --- encryption: the crypt path is invoked (fake rclone), round-trip verifies ---------------------


def test_encrypted_local_roundtrip_via_fake_rclone(tmp_path: Path, monkeypatch):
    fake = _FakeRclone()
    monkeypatch.setattr(rclone, "run_rclone", fake)
    monkeypatch.setattr(engine, "rclone_available", lambda _s: True)  # engine's encryption gate
    monkeypatch.setattr(crypt, "_keyring_get", lambda service, account: "s3cret")  # never the real keychain

    s = _bsettings(tmp_path, backup_encrypt=True)
    art = _make_artifacts(s)

    res = push(s, ("artifacts",))
    assert [r.status for r in res] == ["pushed"]
    assert res[0].location.startswith("local:")

    # the crypt overlay was actually used: obscure was called, and a copy targeted a :crypt: endpoint
    assert any(c[0] == "obscure" for c in fake.calls)
    assert any(c[0] == "copy" and any(":crypt," in a for a in c) for c in fake.calls)
    # manifest records it as encrypted
    m = manifest.load_manifest(s.backup_manifest_path)
    assert m.assets["artifacts"].targets["local"].encrypted is True

    # round-trip restores the exact bytes
    shutil.rmtree(art)
    rres = restore(s, ("artifacts",))
    assert rres[0].status == "restored"
    assert (art / "verbatim.md").read_text() == "VERBATIM one"


@pytest.mark.skipif(shutil.which("rclone") is None, reason="rclone binary not installed")
def test_real_crypt_roundtrip_local(tmp_path: Path, monkeypatch):
    # A genuine rclone crypt round-trip to a local dir (skipped where rclone is absent).
    monkeypatch.setattr(crypt, "_keyring_get", lambda service, account: None)
    monkeypatch.setenv(crypt.PASSPHRASE_ENV, "correct horse battery staple")
    s = _bsettings(tmp_path, backup_encrypt=True)
    art = _make_artifacts(s)

    assert [r.status for r in push(s, ("artifacts",))] == ["pushed"]
    # the on-disk backup is ciphertext: the plaintext must not appear verbatim
    blobs = b"".join(p.read_bytes() for p in (s.backup_local_dir).rglob("*") if p.is_file())
    assert b"LLM transcription two" not in blobs

    shutil.rmtree(art)
    assert restore(s, ("artifacts",))[0].status == "restored"
    assert (art / "llm.md").read_text() == "LLM transcription two"


# --- status --------------------------------------------------------------------------------------


def test_status_reports_targets_and_drift(tmp_path: Path):
    s = _bsettings(tmp_path)
    _make_artifacts(s)
    push(s, ("artifacts",))
    # add a new artifact after the push -> status should report drift
    (s.distilled_dir / "paper1" / "extra.md").write_text("later", encoding="utf-8")

    report = status(s)
    by_asset = {a.asset: a for a in report.assets}
    assert by_asset["artifacts"].n_files == 4
    assert any("extra.md" in a for a in by_asset["artifacts"].drift.added)
    assert not by_asset["artifacts"].drift.clean
    local_detail = next(d for d in by_asset["artifacts"].target_details if d["name"] == "local")
    assert local_detail["available"] is True and local_detail["last_push"] is not None
    # index defaults to none, never pushed
    assert by_asset["index"].targets == ["none"]


# --- rebuild -------------------------------------------------------------------------------------


def test_rebuild_makes_a_fresh_clone_queryable(small_corpus: Settings):
    index_corpus(small_corpus)
    small_corpus.db_path.unlink()  # simulate a fresh clone with no index backup
    assert not small_corpus.db_path.exists()

    summary = rebuild(small_corpus)
    assert len(summary.indexed) == 2 and not summary.failed
    assert small_corpus.db_path.exists()

    con = connect(small_corpus.db_path)
    hits = search_service(con, "signal denoising measured series", k=5, settings=small_corpus)
    assert hits  # queryable again in one command


def test_rebuild_never_runs_live_distillation(small_corpus: Settings, monkeypatch):
    # The expensive transcription must never be re-paid at rebuild: rebuild reuses committed artifacts
    # and otherwise runs deterministic-only — it never invokes the live/heavy distill backend.
    import research_kb.extract.llm as llm_mod

    def boom(*a, **k):
        raise AssertionError("rebuild must not run live distillation")

    monkeypatch.setattr(llm_mod, "run_live_extraction", boom)
    summary = rebuild(small_corpus)  # fresh clone: no DB, no committed llm.md -> deterministic only
    assert len(summary.indexed) == 2 and not summary.failed


# --- offline invariant: index never triggers a backup push ---------------------------------------


def test_index_does_not_trigger_backup(small_corpus: Settings, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(engine, "push", lambda *a, **k: calls.append("push"))
    index_corpus(small_corpus)
    assert calls == []  # index never calls the backup engine
    assert not small_corpus.backup_manifest_path.exists()  # and never writes a manifest
