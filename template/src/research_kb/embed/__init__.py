"""Embedding backends: batch, swappable, eval-arbitrated.

Default is the offline deterministic ``hashing`` embedder (no GPU, reproducible, fine for tests and
for a corpus where BM25 carries most keyword load). ``tei`` targets a text-embeddings-inference
server spun up per indexing run. The active backend follows ``KB_EMBED_BACKEND``.
"""

from __future__ import annotations

from ..config import Settings, get_settings
from .base import EmbeddingProvider
from .hashing import HashingEmbedder
from .tei import TEIEmbedder


def get_query_embedder(con, settings: Settings | None = None) -> EmbeddingProvider:
    """The embedder that matches how this DB was indexed, read from its stored meta.

    Queries must land in the same vector space as the indexed chunks, so the backend/model/dim come
    from the DB (written at index time), not from ambient config — a KB is self-describing and can be
    searched without re-supplying ``KB_EMBED_*``. Falls back to config for a DB with no meta yet.
    """
    from ..db import get_meta

    settings = settings or get_settings()
    backend = get_meta(con, "embed_backend")
    if not backend:
        return get_embedder(settings)
    override = settings.model_copy(
        update={
            "embed_backend": backend,
            "embed_model": get_meta(con, "embed_model") or settings.embed_model,
            "embed_dim": int(get_meta(con, "embed_dim") or settings.embed_dim),
        }
    )
    return get_embedder(override)


def get_embedder(settings: Settings | None = None) -> EmbeddingProvider:
    settings = settings or get_settings()
    if settings.embed_backend == "tei":
        return TEIEmbedder(settings.embed_url, settings.embed_dim, settings.embed_model, settings.embed_batch_size)
    if settings.embed_backend == "fastembed":
        from .fastembed_backend import FastEmbedEmbedder  # lazy: onnxruntime import only when selected

        return FastEmbedEmbedder(settings.embed_model, settings.embed_dim, settings.embed_batch_size)
    if settings.embed_backend == "hashing":
        return HashingEmbedder(settings.embed_dim)
    raise ValueError(f"unknown embed backend: {settings.embed_backend!r}")


__all__ = ["EmbeddingProvider", "HashingEmbedder", "TEIEmbedder", "get_embedder", "get_query_embedder"]
