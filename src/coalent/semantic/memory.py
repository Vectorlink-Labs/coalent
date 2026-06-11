"""Built-in retrievers + a stub synthesizer for trying the semantic cache.

  * ``InMemoryRetriever``   — a tiny in-memory vector store (dev / tests).
  * ``FunctionRetriever``   — wrap any callable as a retriever (escape hatch).
  * ``BaseVectorRetriever`` — base for vector-DB retrievers: implement two methods.
  * ``StubSynthesizer``     — deterministic understanding, no API key.

Swap these for your real backends — the cache itself doesn't change.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Sequence

from .embedding import Embedder, HashingEmbedder, cosine
from .ports import Chunk, Retriever, Synthesis

logger = logging.getLogger(__name__)


class InMemoryRetriever:
    """A mini in-memory vector store: add documents, retrieve top-k by cosine."""

    def __init__(self, *, embedder: Embedder | None = None, top_k: int = 5) -> None:
        self._embedder: Embedder = embedder if embedder is not None else HashingEmbedder()
        self._top_k = top_k
        self._docs: list[tuple[Chunk, tuple[float, ...], str | None]] = []

    def add(
        self, artifact_id: str, text: str, *, namespace: str | None = None, version: str = "v1"
    ) -> None:
        chunk = Chunk(artifact_id=artifact_id, text=text, version=version)
        self._docs.append((chunk, tuple(self._embedder.embed(text)), namespace))

    def retrieve(self, query: str, *, namespace: str | None = None) -> list[Chunk]:
        qe = tuple(self._embedder.embed(query))
        scored: list[tuple[float, Chunk]] = []
        for chunk, emb, doc_ns in self._docs:
            if namespace is not None and doc_ns is not None and doc_ns != namespace:
                continue
            score = cosine(qe, emb)
            if score > 0.0:
                scored.append((score, chunk))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [chunk for _, chunk in scored[: self._top_k]]


class FunctionRetriever:
    """Wrap any ``(query, namespace) -> list[Chunk]`` callable as a Retriever.

    The simplest escape hatch when you already have a search function::

        reality = FunctionRetriever(lambda q, ns: my_search(q))
    """

    def __init__(self, fn: Callable[[str, "str | None"], list[Chunk]]) -> None:
        self._fn = fn

    def retrieve(self, query: str, *, namespace: str | None = None) -> list[Chunk]:
        return self._fn(query, namespace)


class CompositeRetriever:
    """Fan out to several retrievers and merge their evidence into one result.

    Hand this one retriever to a single :class:`SemanticCache` and it builds **one**
    cognition unit per query whose understanding fuses every source (vector search +
    tools + Confluence + …) and whose provenance spans them all — so a change to any
    one source invalidates exactly the units that used it. Duplicate chunks (same
    ``artifact_id`` *and* ``text``) are collapsed; the first occurrence wins, so the
    order of ``retrievers`` is the precedence order.

    Prefer this for a single, unified understanding. If you'd rather keep sources
    isolated, run a separate cache per retriever (or partition one cache by
    ``namespace``) instead — both are valid; this is just the fused option.

    **Fail-open:** a sub-retriever that raises (a flaky tool / API 5xx) is logged
    and skipped — the fused read returns the healthy sources' evidence rather than
    blanking the whole context, honoring the "never return less than RAG" floor.

    Note: give each *logical* source a stable ``artifact_id``; if two sub-retrievers
    return the same ``artifact_id`` with *different* text, both survive (dedup is on
    exact ``artifact_id`` + ``text``), which can blunt skip-no-op invalidation for
    that artifact — prefer distinct ids per source (e.g. ``confluence:hr`` vs
    ``tool:hr-live``).
    """

    def __init__(self, retrievers: Sequence[Retriever]) -> None:
        self._retrievers = list(retrievers)

    def retrieve(self, query: str, *, namespace: str | None = None) -> list[Chunk]:
        merged: list[Chunk] = []
        seen: set[tuple[str, str]] = set()
        for retriever in self._retrievers:
            try:
                chunks = retriever.retrieve(query, namespace=namespace)
            except Exception:  # one flaky source must not blank the fused context
                logger.warning(
                    "CompositeRetriever: %r failed; continuing with other sources",
                    retriever, exc_info=True,
                )
                continue
            for chunk in chunks:
                key = (chunk.artifact_id, chunk.text)
                if key not in seen:
                    seen.add(key)
                    merged.append(chunk)
        return merged


class BaseVectorRetriever(ABC):
    """Base for vector-DB retrievers — implement two short methods and the
    Chunk-mapping boilerplate is handled for you.

    ``search`` returns raw vendor hits (use your client's native API in full);
    ``to_chunk`` maps one hit to a :class:`~coalent.semantic.ports.Chunk` (derive
    ``artifact_id`` from the hit). Return ``None`` from ``to_chunk`` to skip a hit.
    """

    @abstractmethod
    def search(self, query: str, namespace: str | None) -> list[Any]:
        ...

    @abstractmethod
    def to_chunk(self, hit: Any) -> Chunk | None:
        ...

    def retrieve(self, query: str, *, namespace: str | None = None) -> list[Chunk]:
        chunks: list[Chunk] = []
        for hit in self.search(query, namespace):
            chunk = self.to_chunk(hit)
            if chunk is not None and chunk.artifact_id:
                chunks.append(chunk)
        return chunks


class StubSynthesizer:
    """Deterministic, network-free synthesizer producing the structured shape.

    Thin understanding on purpose — the detail lives in the retained raw evidence
    until you swap in :class:`~coalent.semantic.synthesizer.LLMSynthesizer`."""

    def synthesize(self, query: str, chunks: list[Chunk]) -> Synthesis:
        summary = f"[stub] decision-ready understanding for: {query}"
        return Synthesis(
            understanding={"summary": summary, "claims": [], "entities": [], "facts": {}},
            used=list(range(len(chunks))),
        )
