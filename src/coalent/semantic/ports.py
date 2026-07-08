"""Pluggable ports for the semantic cache: retrieval and synthesis.

Deliberately tiny — Coalent is the freshness/reuse layer ABOVE retrieval, so any
retriever (vector DB, GraphRAG, tools, APIs) implements ``Retriever`` and any
model wrapper implements ``Synthesizer``. We never reimplement their search.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Chunk:
    """One piece of retrieved evidence — RETAINED on the unit (the RAG floor)."""

    artifact_id: str            # natural source id, e.g. "confluence:98231"
    text: str
    version: str = ""           # native revision (Confluence version, git sha, ETag…)
    content_hash: str = ""      # optional; the cache hashes text when absent


@runtime_checkable
class Retriever(Protocol):
    """Fetches the evidence relevant to a query. Bring your own backend."""

    def retrieve(self, query: str, *, namespace: str | None = None) -> list[Chunk]:
        ...


@dataclass(frozen=True, slots=True)
class Usage:
    """Token accounting for one LLM call — so the cache's cost AND savings are measurable
    end-to-end (a cache HIT spends zero, which is the whole point). Surfaced on ``Result.usage``
    and aggregated by ``SemanticCache.stats()``."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    cost: float = 0.0           # optional USD, if the provider/caller knows prices

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class Generation:
    """A provider response WITH usage. A provider MAY return this instead of a bare ``str``;
    the synthesizer accepts BOTH (a plain ``str`` -> ``usage=None``), so existing and custom
    providers never break — only the shipped OpenAI/Anthropic adapters opt into reporting usage."""

    text: str
    usage: Usage | None = None


@dataclass(slots=True)
class Synthesis:
    """A synthesizer's output.

    ``used`` lists the indices of the given chunks the synthesis actually relied
    on — they become the unit's PRECISE provenance, so only sources that were
    used can invalidate it. ``ok=False`` means synthesis failed and the caller
    should degrade (keep the raw evidence) rather than cache fabricated content.
    ``usage`` carries the token cost of the synthesis call (``None`` when no LLM
    was called — e.g. the passthrough synthesizer or a usage-less provider).
    """

    understanding: dict[str, Any]
    used: list[int] = field(default_factory=list)
    ok: bool = True
    usage: Usage | None = None


@runtime_checkable
class Synthesizer(Protocol):
    """Turns retrieved chunks into decision-ready understanding."""

    def synthesize(self, query: str, chunks: list[Chunk]) -> Synthesis:
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """A text-in / text-out model (providers.{Stub,OpenAI,Anthropic}Provider).

    ``generate`` may return a plain ``str`` (no usage) OR a :class:`Generation` (text + usage);
    both are accepted, so a custom provider that returns a bare string keeps working.
    """

    def generate(
        self, *, model: str, system: str, user: str, max_tokens: int, temperature: float
    ) -> "str | Generation":
        ...
