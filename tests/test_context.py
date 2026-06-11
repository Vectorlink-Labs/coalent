"""Acceptance test — context intelligence.

Proves: a cache hit that under-covers the query auto-escalates to fresh raw (no
manual signal); a confident hit serves minimum context with raw reachable but not
dumped; the context payload follows the strategy; and minimum-context projection
trims to the query-relevant slice. Plus the drill/widen agent affordances.
"""
from __future__ import annotations

from coalent.semantic import (
    Chunk,
    ContextStrategy,
    InMemoryRetriever,
    SemanticCache,
    StubSynthesizer,
    Synthesis,
)


def test_under_covering_hit_escalates_to_fresh_raw() -> None:
    retriever = InMemoryRetriever(top_k=1)  # one doc per query
    retriever.add("doc:overview", "annual leave policy summary overview")
    retriever.add("doc:table", "leave accrual rollover twentyfive days per year table")
    cache = SemanticCache(retriever, StubSynthesizer(), coverage_floor=0.6)

    cache.get("leave policy")  # builds a unit from doc:overview only (no numbers)

    # A close rephrase that needs the accrual detail the unit never captured.
    result = cache.get("leave policy accrual rollover days")
    assert result.cache_hit is True          # semantic hit on the same unit
    assert result.escalated is True          # but it under-covered -> escalated
    assert "twentyfive" in result.raw_text   # fresh raw pulled the detail in
    assert "twentyfive" in " ".join(result.context["raw"])  # and surfaced in context


def test_confident_hit_serves_minimum_context_raw_reachable() -> None:
    retriever = InMemoryRetriever(top_k=1)
    retriever.add("doc:leave", "leave policy: 21 days of annual leave per year")
    cache = SemanticCache(retriever, StubSynthesizer())

    cache.get("leave policy")
    result = cache.get("leave policy")  # warm, well-covered

    assert result.cache_hit is True
    assert result.escalated is False
    assert result.context["raw"] == []       # context_first: raw not dumped when not needed
    assert "21 days" in result.raw_text       # but the floor is still reachable


def test_minimum_context_projection_trims_to_relevant() -> None:
    retriever = InMemoryRetriever(top_k=1)
    retriever.add("doc:hr", "hr handbook covering leave, payroll and security")

    class RichSynth:
        def synthesize(self, query: str, chunks: list[Chunk]) -> Synthesis:
            return Synthesis(
                understanding={
                    "summary": "HR handbook overview",
                    "claims": ["leave is 21 days", "payroll runs monthly", "badges expire yearly"],
                    "facts": {"annual_leave": "21 days", "payroll": "monthly", "badge": "yearly"},
                },
                used=list(range(len(chunks))),
            )

    cache = SemanticCache(retriever, RichSynth())
    result = cache.get("how much annual leave")

    claims = result.context["understanding"]["claims"]
    assert "leave is 21 days" in claims
    assert "payroll runs monthly" not in claims  # trimmed: irrelevant to the query
    assert "annual_leave" in result.context["understanding"]["facts"]


def test_strategy_controls_raw_payload() -> None:
    retriever = InMemoryRetriever(top_k=1)
    retriever.add("doc:leave", "leave policy: 21 days annual leave")

    raw_cache = SemanticCache(retriever, StubSynthesizer(), strategy=ContextStrategy.CONTEXT_RAW)
    assert "21 days" in " ".join(raw_cache.get("leave policy").context["raw"])

    only_cache = SemanticCache(retriever, StubSynthesizer(), strategy=ContextStrategy.CONTEXT_ONLY)
    assert only_cache.get("leave policy").context["raw"] == []


def test_drill_and_widen_affordances() -> None:
    retriever = InMemoryRetriever(top_k=2)
    retriever.add("doc:leave", "leave policy: 21 days annual leave")
    retriever.add("doc:overview", "hr supports onboarding and leave")
    cache = SemanticCache(retriever, StubSynthesizer())

    result = cache.get("leave policy")
    drilled = cache.drill(result.unit_id)
    assert any("21 days" in chunk.text for chunk in drilled)         # drill -> raw of the unit
    assert cache.drill("nonexistent") == []

    widened = cache.widen("hr onboarding")
    assert any(chunk.artifact_id == "doc:overview" for chunk in widened)  # widen -> fresh retrieval
