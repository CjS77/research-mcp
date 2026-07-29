"""Interactive setup for the backup engine — ``research-kb backup init``.

The backup *engine* (:mod:`research_kb.backup`) only *consumes* config: which targets an asset goes
to, whether pushes are encrypted, and where the crypt passphrase lives. This wizard is the
onboarding layer that *produces* that config. It:

1. **detects** prerequisites — the ``rclone`` binary, existing rclone remotes, an OS keychain;
2. lets the user **choose targets per asset** — cloud (via rclone), ``local``, or "don't backup"
   (defaults: artifacts → a cloud provider, index → don't backup, the ``rebuild`` safety net);
3. **generates the crypt passphrase**, caches it in the OS keychain, and shows it **once** so the
   user can save a recovery copy (K-6 only ever *reads* the key — generating it belongs here);
4. **proves** each chosen target with a real ``push → pull → byte-compare`` round-trip, and refuses
   to persist a target that fails;
5. **persists** the resolved ``KB_BACKUP_*`` config to ``work/backup.env`` so later headless runs
   (``backup push`` / the refresh cron) need no rerun.

Interactive auth stays with the user: authenticating a cloud remote is ``rclone config`` (the user
drives the OAuth), never the wizard — it never handles raw provider secrets. Every rclone call goes
through the engine's :func:`research_kb.backup.rclone.run_rclone` seam and every keychain write
through :func:`research_kb.backup.crypt.set_passphrase`, so this module is fully testable without a
real rclone, keychain, network, or OAuth.
"""

from __future__ import annotations

import contextlib
import json
import secrets
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import click

from ..config import Settings
from . import crypt, rclone
from .crypt import BackupError
from .targets import Transfer, get_target

# A small, current fallback of common rclone cloud backends, shown when ``rclone config providers``
# cannot be queried (rclone absent). rclone supports 70+; these are the ones a KB user most likely
# wants — the live list from rclone is preferred whenever the binary is present.
_STATIC_RCLONE_BACKENDS: tuple[str, ...] = (
    "drive", "dropbox", "onedrive", "protondrive", "s3", "b2", "box",
    "mega", "pcloud", "sftp", "webdav", "swift", "azureblob", "googlecloudstorage",
)

# One CLI answer -> the engine target-name list it maps to (targets are "rclone" | "local" | "none").
_CHOICE_TO_TARGETS: dict[str, tuple[str, ...]] = {
    "cloud": ("rclone",),
    "local": ("local",),
    "both": ("rclone", "local"),
    "none": ("none",),
}

_PROBE_ASSET = "_wizard_probe"  # a throwaway asset name the round-trip proof pushes under


# --- detection -----------------------------------------------------------------------------------


@dataclass
class Detected:
    """What the environment already provides, gathered before any prompt."""

    rclone: bool
    remotes: list[str]
    keychain: bool


def detect(settings: Settings) -> Detected:
    """Probe for rclone, its configured remotes, and a usable OS keychain."""
    have_rclone = rclone.rclone_available(settings)
    return Detected(
        rclone=have_rclone,
        remotes=list_remotes(settings) if have_rclone else [],
        keychain=crypt.keyring_available(),
    )


def list_remotes(settings: Settings) -> list[str]:
    """Configured rclone remote names (``rclone listremotes``), stripped of the trailing colon."""
    if not rclone.rclone_available(settings):
        return []
    try:
        proc = rclone.run_rclone(["listremotes"], settings)
    except Exception:  # noqa: BLE001 — rclone flaky/absent: report "no remotes", never crash setup
        return []
    if proc.returncode != 0:
        return []
    return [ln.strip().rstrip(":") for ln in proc.stdout.splitlines() if ln.strip()]


def rclone_backends(settings: Settings) -> list[str]:
    """Provider names selectable for a new rclone remote — from rclone when present, else the static list."""
    if rclone.rclone_available(settings):
        try:
            proc = rclone.run_rclone(["config", "providers"], settings)
            if proc.returncode == 0 and proc.stdout.strip():
                data = json.loads(proc.stdout)
                names = [p["Name"] for p in data if isinstance(p, dict) and p.get("Name")]
                if names:
                    return sorted(names)
        except Exception:  # noqa: BLE001 — unparseable/old rclone: fall back to the static list
            pass
    return list(_STATIC_RCLONE_BACKENDS)


# --- key generation ------------------------------------------------------------------------------


def generate_passphrase(nbytes: int = 32) -> str:
    """A strong, URL-safe crypt passphrase (cached in the keychain; the user keeps a recovery copy)."""
    return secrets.token_urlsafe(nbytes)


# --- round-trip proof ----------------------------------------------------------------------------


def prove_target(settings: Settings, target_name: str, *, encrypted: bool, passphrase: str | None) -> tuple[bool, str]:
    """Round-trip a random sentinel through a target (push → pull → byte-compare). Touches no asset.

    Returns ``(ok, detail)``. A target that cannot prove a clean round-trip must not be saved as ready.
    """
    target = get_target(target_name)
    if target is None:
        return False, f"unknown target {target_name!r}"
    if not target.available(settings):
        return False, target.unavailable_hint(settings)
    sentinel = secrets.token_hex(16).encode()
    try:
        with tempfile.TemporaryDirectory() as up, tempfile.TemporaryDirectory() as down:
            (Path(up) / "sentinel").write_bytes(sentinel)
            target.push(Transfer(_PROBE_ASSET, Path(up), encrypted, passphrase, settings))
            target.pull(Transfer(_PROBE_ASSET, Path(down), encrypted, passphrase, settings))
            got = (Path(down) / "sentinel").read_bytes()
    except BackupError as exc:
        _cleanup_probe(settings, target_name)
        return False, str(exc)
    _cleanup_probe(settings, target_name)
    if got == sentinel:
        return True, "round-trip verified (push → pull → byte-match)"
    return False, "round-trip mismatch: the sentinel did not survive the encrypt/transfer/decrypt cycle"


def _cleanup_probe(settings: Settings, target_name: str) -> None:
    """Best-effort removal of whatever :func:`prove_target` left at the target."""
    if target_name == "local" and settings.backup_local_dir is not None:
        shutil.rmtree(settings.backup_local_dir / _PROBE_ASSET, ignore_errors=True)
    elif target_name == "rclone" and settings.backup_rclone_remote:
        # cleanup is best-effort; a leftover probe dir is harmless
        with contextlib.suppress(Exception):
            rclone.run_rclone(["purge", f"{settings.backup_rclone_remote.rstrip('/')}/{_PROBE_ASSET}"], settings)


# --- resolved config + persistence ---------------------------------------------------------------


@dataclass
class WizardConfig:
    """The choices the wizard resolves, and the ``KB_BACKUP_*`` env it persists."""

    artifacts_targets: tuple[str, ...] = ("none",)
    index_targets: tuple[str, ...] = ("none",)
    rclone_remote: str = ""
    local_dir: Path | None = None
    encrypt: bool = True
    plaintext_targets: tuple[str, ...] = ()

    def env(self) -> dict[str, str]:
        """The ``KB_BACKUP_*`` assignments to persist. List fields are JSON (how pydantic reads them)."""
        e: dict[str, str] = {
            "KB_BACKUP_ARTIFACTS_TARGETS": json.dumps(list(self.artifacts_targets)),
            "KB_BACKUP_INDEX_TARGETS": json.dumps(list(self.index_targets)),
            "KB_BACKUP_ENCRYPT": "true" if self.encrypt else "false",
            "KB_BACKUP_PLAINTEXT_TARGETS": json.dumps(list(self.plaintext_targets)),
        }
        if self.rclone_remote:
            e["KB_BACKUP_RCLONE_REMOTE"] = self.rclone_remote
        if self.local_dir is not None:
            e["KB_BACKUP_LOCAL_DIR"] = str(self.local_dir)
        return e

    def as_settings(self, settings: Settings) -> Settings:
        """A copy of ``settings`` reflecting these choices — used to prove targets before persisting."""
        return settings.model_copy(update={
            "backup_artifacts_targets": self.artifacts_targets,
            "backup_index_targets": self.index_targets,
            "backup_rclone_remote": self.rclone_remote,
            "backup_local_dir": self.local_dir,
            "backup_encrypt": self.encrypt,
            "backup_plaintext_targets": self.plaintext_targets,
        })


def write_env(env_path: Path, updates: dict[str, str]) -> None:
    """Merge ``KEY=value`` assignments into a dotenv file: replace listed keys, keep every other line."""
    existing = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    keys = set(updates)
    kept = [ln for ln in existing if ln.split("=", 1)[0].strip() not in keys]
    lines = [*kept, *(f"{k}={v}" for k, v in updates.items())]
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prune(targets: tuple[str, ...], good: set[str]) -> tuple[str, ...]:
    """Keep only proven targets (``none`` always survives); collapse to ``('none',)`` if none remain."""
    kept = tuple(t for t in targets if t == "none" or t in good)
    return kept or ("none",)


# --- interactive orchestration -------------------------------------------------------------------


def _configure_remote(settings: Settings, det: Detected) -> str:
    """Resolve the rclone ``remote:path`` — reuse an existing remote or guide the user to create one."""
    if det.remotes and click.confirm("Use an existing rclone remote?", default=True):
        return click.prompt("Remote (name:path)", default=f"{det.remotes[0]}:kb-backup")
    click.echo("Available rclone backends: " + ", ".join(rclone_backends(settings)))
    click.echo("Create + authenticate a remote yourself:  rclone config")
    click.echo("  (you drive the OAuth / enter provider credentials — the wizard never sees them)")
    return click.prompt("Once created, enter the remote (name:path)", default="gdrive:kb-backup")


def _setup_encryption(settings: Settings, det: Detected) -> bool:
    """Turn encryption on (default), generate the passphrase, cache it, and show it once behind a gate."""
    if not det.rclone:
        click.echo("! Encryption needs rclone (its crypt overlay), which is not installed — backups will be PLAINTEXT.")
        click.echo("  Install rclone and re-run `backup init` to enable encryption.")
        return False
    if not click.confirm("Encrypt backups (recommended)?", default=True):
        return False
    passphrase = generate_passphrase()
    cached = crypt.set_passphrase(settings, passphrase)
    click.echo("\n" + "=" * 70)
    click.echo("BACKUP ENCRYPTION PASSPHRASE — shown ONCE.")
    click.echo("Save it in your password manager. WITHOUT IT YOUR BACKUP IS UNRECOVERABLE on a new machine.")
    click.echo(f"\n    {passphrase}\n")
    if cached:
        click.echo(f"Cached in the OS keychain ({settings.backup_keyring_service}/{settings.backup_keyring_account}) "
                   "so push/restore are non-interactive here.")
    else:
        click.echo("! No usable OS keychain — NOT cached. Set KB_BACKUP_PASSPHRASE from the value above for headless use.")
    click.echo("=" * 70)
    while not click.confirm("Type y once you have saved the passphrase somewhere safe", default=False):
        click.echo("Please save it first — it cannot be recovered later.")
    return True


def _prove_and_report(proof_settings: Settings, target_name: str) -> bool:
    """Prove one target's round-trip and print the verdict."""
    encrypted = crypt.is_encrypted(proof_settings, target_name)
    passphrase = crypt.get_passphrase(proof_settings, allow_prompt=False) if encrypted else None
    ok, detail = prove_target(proof_settings, target_name, encrypted=encrypted, passphrase=passphrase)
    click.echo(f"  [{'OK  ' if ok else 'FAIL'}] {target_name}: {detail}")
    return ok


def _print_recovery(cfg: WizardConfig) -> None:
    """Spell out the fresh-machine recovery flow so a laptop loss is survivable."""
    click.echo("\nTo restore on a new machine:")
    if cfg.rclone_remote:
        click.echo("  1. re-auth the rclone remote:  rclone config")
    click.echo(f"  {'2' if cfg.rclone_remote else '1'}. supply the passphrase — from your OS keychain, or "
               "export KB_BACKUP_PASSPHRASE=<the value you saved>")
    click.echo("  → then:  research-kb backup restore")
    if "none" in cfg.index_targets:
        click.echo("  (the index isn't backed up — run `research-kb rebuild` to reconstruct it from the artifacts)")


def run_wizard(settings: Settings, env_path: Path) -> WizardConfig:
    """Drive the interactive setup and persist the resolved config to ``env_path``. Returns the config."""
    det = detect(settings)
    click.echo("=== research-kb backup setup ===")
    click.echo(f"rclone: {'found' if det.rclone else 'NOT found — install it for cloud backup + encryption'}")
    if det.remotes:
        click.echo(f"existing rclone remotes: {', '.join(det.remotes)}")
    click.echo(f"OS keychain: {'available' if det.keychain else 'unavailable (passphrase falls back to env/prompt)'}\n")

    cfg = WizardConfig()
    cfg.artifacts_targets = _CHOICE_TO_TARGETS[
        click.prompt("Back up distillation ARTIFACTS to", type=click.Choice(list(_CHOICE_TO_TARGETS)), default="cloud")
    ]
    cfg.index_targets = _CHOICE_TO_TARGETS[
        click.prompt("Back up the INDEX to (rebuild is its safety net)",
                     type=click.Choice(list(_CHOICE_TO_TARGETS)), default="none")
    ]

    uses = set(cfg.artifacts_targets) | set(cfg.index_targets)
    if "rclone" in uses:
        cfg.rclone_remote = _configure_remote(settings, det)
    if "local" in uses:
        default_dir = str(settings.distilled_dir.parent / "backup")
        cfg.local_dir = Path(click.prompt("Local backup directory", default=default_dir))

    active = uses - {"none"}
    cfg.encrypt = _setup_encryption(settings, det) if active else False

    if active:
        click.echo("\nProving each target with a real push → pull → verify round-trip:")
        proof_settings = cfg.as_settings(settings)
        good = {t for t in sorted(active) if _prove_and_report(proof_settings, t)}
        cfg.artifacts_targets = _prune(cfg.artifacts_targets, good)
        cfg.index_targets = _prune(cfg.index_targets, good)

    write_env(env_path, cfg.env())
    click.echo(f"\nSaved backup config to {env_path}")
    click.echo("`research-kb backup push` and the refresh cron read it automatically — no rerun needed.")
    _print_recovery(cfg)
    return cfg
