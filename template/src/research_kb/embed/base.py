"""Embedding provider interface. Providers return unit vectors of a fixed dimension."""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """A batch embedder. ``dim`` must equal the DB's ``vec_chunks`` dimension."""

    name: str
    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into ``dim``-length vectors (one per input, same order)."""

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        """Embed search queries. Defaults to :meth:`embed`; providers with an asymmetric query encoder
        (e.g. a bge instruction prefix) override this so queries and passages are encoded to match."""
        return self.embed(texts)
