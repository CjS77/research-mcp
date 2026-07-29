"""In-process ONNX embedding backend via ``fastembed``.

A true neural embedder that needs no GPU, no server, and no torch: fastembed runs a quantized ONNX
model on CPU and downloads it from the HF hub once. This is the runs-anywhere alternative to the TEI
HTTP backend. bge-family models use an asymmetric encoder, so queries go through :meth:`embed_query`
(which prepends the model's retrieval instruction) while passages use :meth:`embed`.
"""

from __future__ import annotations

import os
from pathlib import Path

from .base import EmbeddingProvider


class FastEmbedEmbedder(EmbeddingProvider):
    def __init__(self, model: str, dim: int, batch_size: int = 64) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # fail loudly — never silently fall back to a weaker embedder
            raise RuntimeError(
                "embed backend 'fastembed' is selected but the `fastembed` package is not installed. "
                "Install it (`uv sync` or `uv pip install fastembed`), or set "
                "KB_EMBED_BACKEND=hashing to use the offline embedder."
            ) from exc

        self.name = model
        self.dim = dim
        self.batch_size = batch_size
        # fastembed's own default cache is $TMPDIR/fastembed_cache, which is wiped on reboot and
        # forces a ~210 MB model re-download; default to a persistent per-user cache instead.
        cache_dir = os.environ.get("FASTEMBED_CACHE_PATH") or str(Path.home() / ".cache" / "fastembed")
        try:
            self._model = TextEmbedding(model_name=model, cache_dir=cache_dir)
        except Exception as exc:  # missing/unknown model, or HF hub unreachable — say so, don't degrade
            raise RuntimeError(
                f"fastembed could not load the embedding model {model!r}: {exc}. Check KB_EMBED_MODEL "
                "is a supported fastembed model and the HuggingFace hub is reachable to download it, "
                "or set KB_EMBED_BACKEND=hashing to use the offline embedder."
            ) from exc

    def _as_vectors(self, raw) -> list[list[float]]:
        out: list[list[float]] = []
        for v in raw:
            vec = [float(x) for x in v]
            if len(vec) != self.dim:
                raise ValueError(
                    f"fastembed model {self.name!r} returned dim={len(vec)} but the DB expects {self.dim}; "
                    "re-init the DB (set KB_EMBED_DIM to match the model)."
                )
            out.append(vec)
        return out

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._as_vectors(self._model.embed(texts, batch_size=self.batch_size))

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        return self._as_vectors(self._model.query_embed(texts, batch_size=self.batch_size))
