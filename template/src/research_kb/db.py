"""SQLite connection management: sqlite-vec loading, schema init, vector (de)serialization.

The database is a single file (sqlite-vec + FTS5, no dedicated vector DB at this
scale). ``vec_chunks`` is created here rather than in ``schema.sql`` so its
dimension follows ``KB_EMBED_DIM``; the chosen embedding provenance is recorded in ``kb_meta``
so a later run with an incompatible dimension fails loudly instead of corrupting search.
"""

from __future__ import annotations

import sqlite3
import struct
from importlib import resources
from pathlib import Path

import sqlite_vec

from .config import Settings, get_settings


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a connection with sqlite-vec loaded, WAL journaling, and row access by name."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA journal_mode = WAL")
    con.execute("PRAGMA synchronous = NORMAL")
    return con


def _load_schema_sql() -> str:
    return resources.files("research_kb").joinpath("schema.sql").read_text(encoding="utf-8")


def init_db(settings: Settings | None = None, con: sqlite3.Connection | None = None) -> sqlite3.Connection:
    """Create the schema, the dimension-specific vec table, and record embedding provenance."""
    settings = settings or get_settings()
    con = con or connect(settings.db_path)
    con.executescript(_load_schema_sql())

    _init_meta(con, settings)
    dim_str = get_meta(con, "embed_dim")
    assert dim_str is not None  # just written by _init_meta above
    dim = int(dim_str)
    con.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
        f"id INTEGER PRIMARY KEY, embedding float[{dim}])"
    )
    con.commit()
    return con


def _init_meta(con: sqlite3.Connection, settings: Settings) -> None:
    con.execute("CREATE TABLE IF NOT EXISTS kb_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    existing_dim = get_meta(con, "embed_dim")
    if existing_dim is None:
        set_meta(con, "embed_dim", str(settings.embed_dim))
        set_meta(con, "embed_backend", settings.embed_backend)
        set_meta(con, "embed_model", settings.embed_model)
        set_meta(con, "schema_version", "1")
    elif int(existing_dim) != settings.embed_dim:
        raise RuntimeError(
            f"embedding dimension mismatch: DB was built at dim={existing_dim} but config is "
            f"{settings.embed_dim}. Re-init the DB (delete {settings.db_path}) to switch dimension."
        )


def get_meta(con: sqlite3.Connection, key: str) -> str | None:
    row = con.execute("SELECT value FROM kb_meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def set_meta(con: sqlite3.Connection, key: str, value: str) -> None:
    con.execute(
        "INSERT INTO kb_meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def pack_vector(values: list[float]) -> bytes:
    """Serialize a float vector to the little-endian float32 blob sqlite-vec expects."""
    return struct.pack(f"<{len(values)}f", *values)


def unpack_vector(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))
