"""text-embeddings-inference (TEI) HTTP backend.

Batch, not a standing service: spin TEI up for an indexing run, tear it down after. This client just
POSTs to ``/embed``. Dimension is asserted against the DB's configured dimension so a model swap
without a re-init fails loudly.
"""

from __future__ import annotations

import httpx

from .base import EmbeddingProvider


class TEIEmbedder(EmbeddingProvider):
    def __init__(self, url: str, dim: int, model: str = "tei", batch_size: int = 64, timeout: float = 60.0) -> None:
        self.url = url.rstrip("/")
        self.dim = dim
        self.name = model
        self.batch_size = batch_size
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        with httpx.Client(timeout=self.timeout) as client:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                resp = client.post(f"{self.url}/embed", json={"inputs": batch, "truncate": True})
                resp.raise_for_status()
                vectors = resp.json()
                for v in vectors:
                    if len(v) != self.dim:
                        raise ValueError(
                            f"TEI returned dim={len(v)} but DB expects {self.dim}; re-init the DB for this model."
                        )
                    out.append([float(x) for x in v])
        return out
