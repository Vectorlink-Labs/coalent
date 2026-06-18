"""Embeddings for the semantic cache.

The cache is keyed by query *meaning*, so we need an embedder. The ``Embedder``
port lets you plug any model. ``OpenAIEmbedder`` gives real semantic embeddings;
``FunctionEmbedder`` wraps any callable (e.g. a local sentence-transformers model);
``HashingEmbedder`` is a deterministic, dependency-free bag-of-words used as a
no-key fallback (it matches keyword overlap, NOT meaning). ``default_embedder``
picks the best available automatically.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import warnings
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

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


class OpenAIEmbedder:
    """Real semantic embeddings via the OpenAI Embeddings API (extra ``coalent[openai]``).

    Bring your own ``client`` (a ``openai.OpenAI``) so we never pin a version, or let
    it construct one from ``api_key`` / ``OPENAI_API_KEY``. ``text-embedding-3-small``
    is an excellent, cheap default; ``text-embedding-3-large`` for maximum accuracy.
    """

    def __init__(
        self, model: str = "text-embedding-3-small", *, client: Any = None, api_key: str | None = None
    ) -> None:
        if client is None:
            from openai import OpenAI  # lazy: only needed when actually used

            client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self._client = client
        self._model = model

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(model=self._model, input=text)
        return list(response.data[0].embedding)


class FunctionEmbedder:
    """Wrap any ``text -> sequence[float]`` callable as an Embedder.

    The escape hatch for local/self-hosted models::

        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("all-MiniLM-L6-v2")
        embedder = FunctionEmbedder(lambda t: m.encode(t).tolist())
    """

    def __init__(self, fn: Callable[[str], "Sequence[float]"]) -> None:
        self._fn = fn

    def embed(self, text: str) -> list[float]:
        return list(self._fn(text))


def default_embedder() -> "Embedder":
    """The best embedder available without explicit config.

    Uses ``OpenAIEmbedder`` when the ``openai`` package is installed AND
    ``OPENAI_API_KEY`` is set (accurate, semantic). Otherwise falls back to the
    lexical ``HashingEmbedder`` and WARNS — because keyword-overlap matching will
    miss semantically-similar queries. Always override with ``embedder=``.
    """
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAIEmbedder()
        except ImportError:
            warnings.warn(
                "coalent: OPENAI_API_KEY is set but the 'openai' package isn't installed — "
                "run `pip install coalent[openai]` for semantic embeddings. "
                "Falling back to the lexical HashingEmbedder for now.",
                stacklevel=2,
            )
            return HashingEmbedder()
    warnings.warn(
        "coalent: no `embedder` was provided and OPENAI_API_KEY is not set, so the cache is using "
        "HashingEmbedder — which matches on KEYWORD OVERLAP, not meaning, so semantically-similar "
        "queries can miss the cache. For accurate semantic hits: `pip install coalent[openai]` and set "
        "OPENAI_API_KEY, or pass embedder=YourEmbedder(...). See https://coalent.ai/docs/retriever.",
        stacklevel=2,
    )
    return HashingEmbedder()


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two equal-length vectors (0.0 on a zero vector)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
