"""Deterministic feature-hashing embedder — the offline default.

No model, no GPU, no network: a signed hashing-trick bag-of-(uni+bi)grams, L2-normalized. It gives a
weak but real lexical-overlap signal (and reproducible tests). For normalized vectors, sqlite-vec's
L2 knn ranks identically to cosine. Swap to ``tei`` for a true semantic model.
"""

from __future__ import annotations

import hashlib
import math
import re

from .base import EmbeddingProvider

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    words = _TOKEN_RE.findall(text.lower())
    bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]
    return words + bigrams


class HashingEmbedder(EmbeddingProvider):
    def __init__(self, dim: int = 768) -> None:
        self.dim = dim
        self.name = "hashing-v1"

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _tokens(text):
            h = int.from_bytes(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest(), "little")
            idx = h % self.dim
            vec[idx] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]
