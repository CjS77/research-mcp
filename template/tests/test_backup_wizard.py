"""Backup-setup wizard (`research-kb backup init`): detect, choose, key-gen, prove, persist.

No test touches a real rclone binary, OS keychain, network, or OAuth. The local plaintext round-trip
runs for real (tmp dir); the encrypted path is exercised through a fake rclone (crypt modelled as a
passthrough copy) and a fake keychain (a dict behind the crypt seams).
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import click
from click.testing import CliRunner

from research_kb.backup import crypt, wizard
from research_kb.cli import main
from research_kb.config import Settings

# --- helpers ------------------------------------------------------------------------------------


def _wsettings(tmp_path: Path, **over) -> Settings:
    kw: dict = dict(
        db_path=tmp_path / "work" / "data" / "kb.sqlite",
        distilled_dir=tmp_path / "work" / "distilled",
        base_dir=tmp_path,
        reference_dir=tmp_path / "ref",
        docs_dir=tmp_path / "docs",
        embed_backend="hashing",
        embed_dim=256,
    )
    kw.update(over)
    return Settings(**kw)


class _FakeRclone:
    """Stand-in for the ``rclone`` binary: ``obscure`` + ``copy`` (crypt = passthrough copy)."""

    def __init__(self, providers: str | None = None, remotes: str = "") -> None:
        self.calls: list[list[str]] = []
        self._providers = providers
        self._remotes = remotes

    def __call__(self, args: list[str], settings: Settings):  # noqa: ANN204 — mimics run_rclone
        self.calls.append(list(args))
        if args[:2] == ["config", "providers"]:
            return SimpleNamespace(returncode=0, stdout=self._providers or "[]", stderr="")
        if args[0] == "listremotes":
            return SimpleNamespace(returncode=0, stdout=self._remotes, stderr="")
        if args[0] == "obscure":
            return SimpleNamespace(returncode=0, stdout=f"OBSCURED::{args[1]}", stderr="")
        if args[0] == "copy":
            src, dst = self._real(args[1]), self._real(args[2])
            Path(dst).mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst, dirs_exist_ok=True)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    @staticmethod
    def _real(arg: str) -> str:
        m = re.match(r":crypt,remote='([^']*)',", arg)
        return m.group(1) if m else arg


def _fake_keychain(monkeypatch) -> dict:
    store: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(crypt, "_keyring_set", lambda svc, acc, sec: bool(store.__setitem__((svc, acc), sec)) or True)
    monkeypatch.setattr(crypt, "_keyring_get", lambda svc, acc: store.get((svc, acc)))
    monkeypatch.setattr(crypt, "keyring_available", lambda: True)
    return store


def _drive(settings: Settings, env_path: Path, keys: str):
    """Run the wizard under CliRunner, feeding ``keys`` to its prompts."""

    @click.command()
    def cmd() -> None:
        wizard.run_wizard(settings, env_path)

    return CliRunner().invoke(cmd, input=keys)


# --- detect / discovery -------------------------------------------------------------------------


def test_detect_reports_rclone_remotes_keychain(tmp_path, monkeypatch):
    monkeypatch.setattr(wizard.rclone, "rclone_available", lambda _s: True)
    monkeypatch.setattr(wizard.rclone, "run_rclone", _FakeRclone(remotes="gdrive:\ns3:\n"))
    monkeypatch.setattr(wizard.crypt, "keyring_available", lambda: True)
    det = wizard.detect(_wsettings(tmp_path))
    assert det.rclone is True and det.keychain is True
    assert det.remotes == ["gdrive", "s3"]


def test_list_remotes_empty_without_rclone(tmp_path, monkeypatch):
    monkeypatch.setattr(wizard.rclone, "rclone_available", lambda _s: False)
    assert wizard.list_remotes(_wsettings(tmp_path)) == []


def test_rclone_backends_from_providers(tmp_path, monkeypatch):
    monkeypatch.setattr(wizard.rclone, "rclone_available", lambda _s: True)
    monkeypatch.setattr(wizard.rclone, "run_rclone",
                        _FakeRclone(providers='[{"Name":"s3"},{"Name":"drive"},{"Name":"dropbox"}]'))
    assert wizard.rclone_backends(_wsettings(tmp_path)) == ["drive", "dropbox", "s3"]


def test_rclone_backends_static_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(wizard.rclone, "rclone_available", lambda _s: False)
    backends = wizard.rclone_backends(_wsettings(tmp_path))
    assert "drive" in backends and "dropbox" in backends and "s3" in backends


def test_generate_passphrase_is_strong_and_urlsafe():
    p = wizard.generate_passphrase()
    assert len(p) >= 32 and re.fullmatch(r"[A-Za-z0-9_-]+", p)
    assert wizard.generate_passphrase() != wizard.generate_passphrase()


# --- config encoding + persistence --------------------------------------------------------------


def test_wizardconfig_env_encoding():
    cfg = wizard.WizardConfig(
        artifacts_targets=("rclone", "local"), index_targets=("none",),
        rclone_remote="gdrive:kb", local_dir=Path("/b"), encrypt=True,
    )
    env = cfg.env()
    assert env["KB_BACKUP_ARTIFACTS_TARGETS"] == '["rclone", "local"]'
    assert env["KB_BACKUP_INDEX_TARGETS"] == '["none"]'
    assert env["KB_BACKUP_ENCRYPT"] == "true"
    assert env["KB_BACKUP_RCLONE_REMOTE"] == "gdrive:kb" and env["KB_BACKUP_LOCAL_DIR"] == "/b"


def test_write_env_replaces_keys_keeps_others(tmp_path):
    env_path = tmp_path / "backup.env"
    env_path.write_text("KB_OTHER=keep\nKB_BACKUP_ENCRYPT=true\n", encoding="utf-8")
    wizard.write_env(env_path, {"KB_BACKUP_ENCRYPT": "false", "KB_BACKUP_LOCAL_DIR": "/d"})
    lines = env_path.read_text(encoding="utf-8").splitlines()
    assert "KB_OTHER=keep" in lines
    assert "KB_BACKUP_ENCRYPT=false" in lines and "KB_BACKUP_ENCRYPT=true" not in lines
    assert "KB_BACKUP_LOCAL_DIR=/d" in lines


# --- prove_target -------------------------------------------------------------------------------


def test_prove_target_local_plaintext_roundtrip_and_cleanup(tmp_path):
    s = _wsettings(tmp_path, backup_local_dir=tmp_path / "bak", backup_encrypt=False)
    ok, detail = wizard.prove_target(s, "local", encrypted=False, passphrase=None)
    assert ok is True and "verified" in detail
    assert not (s.backup_local_dir / wizard._PROBE_ASSET).exists()  # probe cleaned up


def test_prove_target_unavailable(tmp_path):
    s = _wsettings(tmp_path, backup_local_dir=None)  # local dir not configured
    ok, detail = wizard.prove_target(s, "local", encrypted=False, passphrase=None)
    assert ok is False and "KB_BACKUP_LOCAL_DIR" in detail


def test_prove_target_encrypted_via_fake_rclone(tmp_path, monkeypatch):
    monkeypatch.setattr(wizard.rclone, "run_rclone", _FakeRclone())
    s = _wsettings(tmp_path, backup_local_dir=tmp_path / "bak", backup_encrypt=True)
    ok, _ = wizard.prove_target(s, "local", encrypted=True, passphrase="s3cret")
    assert ok is True


# --- full wizard: local plaintext (rclone absent), config reloadable ----------------------------


def test_wizard_local_plaintext_persists_and_reloads(tmp_path, monkeypatch):
    monkeypatch.setattr(wizard.rclone, "rclone_available", lambda _s: False)  # no rclone -> plaintext
    monkeypatch.setattr(wizard.crypt, "keyring_available", lambda: False)
    s = _wsettings(tmp_path)
    env_path = tmp_path / "work" / "backup.env"

    # artifacts -> local ; index -> none ; local dir -> default (empty line)
    res = _drive(s, env_path, "local\nnone\n\n")
    assert res.exit_code == 0, res.output
    assert "PLAINTEXT" in res.output and "[OK  ] local" in res.output
    assert env_path.exists()

    reloaded = Settings(_env_file=str(env_path), base_dir=tmp_path, distilled_dir=tmp_path / "work" / "distilled",
                        db_path=tmp_path / "work" / "data" / "kb.sqlite", reference_dir=tmp_path / "ref",
                        docs_dir=tmp_path / "docs", embed_backend="hashing", embed_dim=256)
    assert reloaded.backup_artifacts_targets == ("local",)
    assert reloaded.backup_index_targets == ("none",)  # "don't backup" honored
    assert reloaded.backup_encrypt is False
    assert reloaded.backup_local_dir == tmp_path / "work" / "backup"  # default = distilled_dir.parent / "backup"


def test_wizard_encryption_generates_key_shown_once_and_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(wizard.rclone, "rclone_available", lambda _s: True)
    monkeypatch.setattr(wizard.rclone, "run_rclone", _FakeRclone())
    store = _fake_keychain(monkeypatch)
    s = _wsettings(tmp_path)
    env_path = tmp_path / "work" / "backup.env"

    # artifacts -> local ; index -> none ; local dir default ; encrypt? y ; saved? y
    res = _drive(s, env_path, "local\nnone\n\ny\ny\n")
    assert res.exit_code == 0, res.output
    assert "PASSPHRASE" in res.output and "shown ONCE" in res.output
    # the generated passphrase was cached in the (fake) keychain, exactly once
    assert list(store.keys()) == [(s.backup_keyring_service, s.backup_keyring_account)]
    saved = store[(s.backup_keyring_service, s.backup_keyring_account)]
    assert saved and saved in res.output  # the shown value is the cached value
    assert "KB_BACKUP_ENCRYPT=true" in env_path.read_text(encoding="utf-8")


def test_wizard_failed_roundtrip_not_persisted(tmp_path, monkeypatch):
    monkeypatch.setattr(wizard.rclone, "rclone_available", lambda _s: False)
    monkeypatch.setattr(wizard.crypt, "keyring_available", lambda: False)
    monkeypatch.setattr(wizard, "prove_target", lambda *a, **k: (False, "boom"))  # every target fails to prove
    s = _wsettings(tmp_path)
    env_path = tmp_path / "work" / "backup.env"

    res = _drive(s, env_path, "local\nnone\n\n")
    assert res.exit_code == 0, res.output
    assert "[FAIL] local" in res.output
    # a target that could not prove a round-trip is not saved as ready -> collapses to "none"
    assert 'KB_BACKUP_ARTIFACTS_TARGETS=["none"]' in env_path.read_text(encoding="utf-8")


def test_backup_init_wired_into_cli():
    res = CliRunner().invoke(main, ["backup", "init", "--help"])
    assert res.exit_code == 0
    assert "targets" in res.output.lower()
