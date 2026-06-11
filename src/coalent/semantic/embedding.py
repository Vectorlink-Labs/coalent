"""Embeddings for the semantic cache.

The cache is keyed by query *meaning*, so we need an embedder. The ``Embedder``
port lets you plug any model (OpenAI, local, etc.). ``HashingEmbedder`` is a
deterministic, dependency-free default — a normalized bag-of-words over hashed
tokens — good enough for tests, demos, and keyword-ish similarity without an API
key. Swap it for a real semantic embedder in production.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, Sequence, runtime_checkable

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an the is are was were be been being am of for to in on at by with from as "
    "and or but if then else this that these those it its we you i he she they them "
    "our your their my me us what which who whom how why when where give show tell "
    "do does did about some more please need want can could would should".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, drop stopwords/short tokens, and crudely singularize."""
    out: list[str] = []
    for token in _TOKEN.findall(text.lower()):
        if token in _STOP or len(token) < 2:
            continue
        if len(token) > 3 and token.endswith("s"):
            token = token[:-1]  # crude stem so leave/leaves, policy/policies align
        out.append(token)
    return out


@runtime_checkable
class Embedder(Protocol):
    """Turns text into a vector. The only contract the semantic cache needs."""

    def embed(self, text: str) -> list[float]:
        ...


class HashingEmbedder:
    """Deterministic, dependency-free bag-of-words embedder (default)."""

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for token in tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self._dim
            vec[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vec))
        if norm == 0.0:
            return vec
        return [value / norm for value in vec]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two equal-length vectors (0.0 on a zero vector)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
