"""Phase B/D — the relevance_gate retriever-precision hook + the depth knob.

Proves:
  1. relevance_gate drops irrelevant chunks BEFORE synthesis, so they never enter the
     understanding/provenance OR the raw floor (the "5 retrieved, 3 relevant" fix that
     keeps Coalent retriever-agnostic — BYO filter, no dep).
  2. The depth knob appends a tier directive to the synthesizer instruction (deeper =
     preserve every detail; terser = cheaper).
"""
from __future__ import annotations

from coalent import Chunk, FunctionRetriever, LLMSynthesizer, SemanticCache, StubSynthesizer
from coalent.semantic.synthesizer import _depth_directive


def _noisy_retriever() -> FunctionRetriever:
    def search(query: str, namespace: str | None = None) -> list[Chunk]:
        return [
            Chunk("doc:relevant", "leave policy: 21 days of annual leave per year"),
            Chunk("doc:noise", "cafeteria menu and parking information"),
        ]

    return FunctionRetriever(search)


def test_relevance_gate_filters_before_synthesis() -> None:
    def gate(query: str, chunks: list[Chunk]) -> list[Chunk]:
        return [c for c in chunks if c.artifact_id == "doc:relevant"]

    cache = SemanticCache(_noisy_retriever(), StubSynthesizer(), relevance_gate=gate)
    result = cache.get("leave policy")

    # the noise chunk never entered the unit -> changing it matches nothing
    assert cache.source_changed("doc:noise", text="changed").matched_units == 0
    # the relevant chunk IS provenance -> changing it dirties the unit
    assert result.unit_id in cache.source_changed("doc:relevant", text="now 25 days").dirtied


def test_relevance_gate_cleans_the_raw_floor() -> None:
    def gate(query: str, chunks: list[Chunk]) -> list[Chunk]:
        return [c for c in chunks if c.artifact_id == "doc:relevant"]

    # CONTEXT_RAW so the floor is shipped — the gated noise must NOT appear.
    cache = SemanticCache(
        _noisy_retriever(), StubSynthesizer(), relevance_gate=gate, strategy="context_raw"
    )
    raw = " ".join(cache.get("leave policy").context["raw"])
    assert "annual leave" in raw       # kept
    assert "cafeteria" not in raw      # dropped before it ever became evidence


def test_no_gate_keeps_all_chunks() -> None:
    cache = SemanticCache(_noisy_retriever(), StubSynthesizer(), strategy="context_raw")
    raw = " ".join(cache.get("leave policy").context["raw"])
    assert "annual leave" in raw and "cafeteria" in raw  # default: nothing filtered


class _RecordingProvider:
    def __init__(self) -> None:
        self.user = ""

    def generate(self, *, model: str, system: str, user: str, max_tokens: int, temperature: float):  # type: ignore[no-untyped-def]
        self.user = user
        return '{"summary": "ok", "claims": [], "used": [0]}'


def test_depth_directive_tiers() -> None:
    assert "TERSE" in _depth_directive(0.0)
    assert _depth_directive(0.5) == ""        # mid: default balance, no suffix
    assert "EXHAUSTIVE" in _depth_directive(1.0)


def test_depth_appends_directive_to_synthesis_prompt() -> None:
    chunk = [Chunk("a", "some source text")]

    deep = _RecordingProvider()
    LLMSynthesizer(deep, depth=1.0).synthesize("q", chunk)
    assert "EXHAUSTIVE" in deep.user

    terse = _RecordingProvider()
    LLMSynthesizer(terse, depth=0.0).synthesize("q", chunk)
    assert "TERSE" in terse.user

    mid = _RecordingProvider()
    LLMSynthesizer(mid, depth=0.5).synthesize("q", chunk)
    assert "TERSE" not in mid.user and "EXHAUSTIVE" not in mid.user
