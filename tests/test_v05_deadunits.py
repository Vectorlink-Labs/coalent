"""v0.5 — the M4-discovered dead-unit poison, pinned (hermetic).

96/609 real-corpus units had failed synthesis and were cached HOLLOW; the coverage gate
reported a vacuous 1.0 on them, suppressing recall AND the RAG floor, and every matching
read served empty context. Pins: (1) an empty unit reports coverage 0.0 (recall/escalation
can fire); (2) a _synthesis_failed unit self-heals — re-materializes on its next matching
read instead of serving hollow; (3) passthrough/structured units with real content KEEP
the benign vacuous-1.0 (don't penalize what can't be judged)."""
from __future__ import annotations

from typing import Any

from coalent.semantic import Chunk, InMemoryRetriever, SemanticCache, Synthesis


class FlakySynth:
    """Fails the first synthesis (like a timeout-truncated JSON), succeeds after."""

    def __init__(self) -> None:
        self.calls = 0

    def synthesize(self, query: str, chunks: list[Chunk]) -> Synthesis:
        self.calls += 1
        if self.calls == 1:
            return Synthesis(understanding={"_synthesis_failed": True}, used=[], ok=False)
        text = " ".join(c.text for c in chunks)
        return Synthesis(understanding={"summary": text, "claims": [text]},
                         used=list(range(len(chunks))))


def _world() -> tuple[SemanticCache, FlakySynth]:
    retriever = InMemoryRetriever()
    retriever.add("hr:leave", "Annual leave is 21 days.")
    synth = FlakySynth()
    cache = SemanticCache(retriever, synth, hit_threshold=0.1, coverage_floor=0.0)
    return cache, synth


def test_empty_unit_reports_zero_coverage() -> None:
    cache, _synth = _world()
    r1 = cache.get("annual leave days")            # first build FAILS -> hollow unit
    unit = cache._units[r1.unit_id]
    assert unit.understanding.get("_synthesis_failed")
    assert cache._semantic_coverage((1.0, 0.0), unit) == 0.0   # was the vacuous 1.0


def test_failed_unit_self_heals_on_next_read() -> None:
    cache, synth = _world()
    r1 = cache.get("annual leave days")
    assert r1.cache_hit is False and synth.calls == 1
    r2 = cache.get("annual leave days")            # matches the hollow unit -> re-materialize
    assert r2.cache_hit is False                   # healed via rebuild, not served hollow
    assert synth.calls == 2
    assert not cache._units[r2.unit_id].understanding.get("_synthesis_failed")
    assert "21 days" in str(cache._units[r2.unit_id].understanding)
    r3 = cache.get("annual leave days")            # now a REAL hit
    assert r3.cache_hit is True and synth.calls == 2


def test_passthrough_units_keep_vacuous_coverage() -> None:
    cache, _synth = _world()
    cache.get("annual leave days")                 # hollow (first synth fails)
    r2 = cache.get("annual leave days")            # healed -> has summary + claims
    unit: Any = cache._units[r2.unit_id]
    try:
        unit.claim_embeddings = ()                 # content but NO claim embeddings
    except AttributeError:                          # frozen/slots dataclass
        object.__setattr__(unit, "claim_embeddings", ())
    assert cache._semantic_coverage((1.0,), unit) == 1.0   # benign vacuous-1.0 preserved
