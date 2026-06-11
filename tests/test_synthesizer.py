"""Acceptance test — the structured, citation-grounded synthesizer.

Proves:
  1. structured understanding (summary/claims/entities/facts) from JSON.
  2. PRECISE provenance — only the cited source can invalidate the unit
     (a change to a retrieved-but-uncited source touches nothing). The moat.
  3. no-garbage degrade — a parse failure never caches fabricated/raw text as
     understanding, but the raw evidence (RAG floor) is still returned.
  4. citation fallback — valid JSON with no "used" depends on all sources, flagged.
"""
from __future__ import annotations

import json

from coalent import StubProvider
from coalent.semantic import (
    Chunk,
    InMemoryRetriever,
    LLMSynthesizer,
    SemanticCache,
    Synthesis,
)

HANDBOOK = "confluence:hr-handbook"
OVERVIEW = "confluence:hr-overview"


def _chunks() -> list[Chunk]:
    return [
        Chunk(HANDBOOK, "Leave policy: 21 days of annual leave per year.", version="v1"),
        Chunk(OVERVIEW, "HR supports onboarding, payroll and leave.", version="v1"),
    ]


def test_structured_understanding_from_citations() -> None:
    canned = json.dumps(
        {
            "summary": "Full-time staff get 21 days annual leave.",
            "claims": ["21 days annual leave"],
            "entities": ["annual leave"],
            "facts": {"annual_leave": "21 days"},
            "used": [0],
        }
    )
    synth = LLMSynthesizer(StubProvider(canned=canned))
    result = synth.synthesize("leave policy?", _chunks())

    assert result.ok is True
    assert result.used == [0]
    assert result.understanding["facts"]["annual_leave"] == "21 days"
    assert "used" not in result.understanding  # citation key stripped from understanding


def test_only_cited_source_invalidates() -> None:
    retriever = InMemoryRetriever()
    retriever.add(HANDBOOK, "Leave policy: 21 days annual leave per year.")
    retriever.add(OVERVIEW, "HR supports onboarding, payroll and leave.")

    # A synthesizer that cites ONLY the handbook (deterministic, by artifact id).
    class CiteHandbook:
        def synthesize(self, query: str, chunks: list[Chunk]) -> Synthesis:
            used = [i for i, c in enumerate(chunks) if c.artifact_id == HANDBOOK]
            return Synthesis(understanding={"summary": "x"}, used=used)

    cache = SemanticCache(retriever, CiteHandbook())
    result = cache.get("leave policy details")

    # Change to a retrieved-but-UNCITED source must touch nothing (precision).
    assert cache.source_changed(OVERVIEW, text="totally changed").dirtied == []
    assert cache.get("leave policy details").cache_hit is True

    # Change to the cited source dirties exactly this unit.
    changed = cache.source_changed(HANDBOOK, text="Leave policy: now 25 days.")
    assert result.unit_id in changed.dirtied


def test_parse_failure_degrades_without_garbage() -> None:
    synth = LLMSynthesizer(StubProvider(canned="totally not json"))
    result = synth.synthesize("leave?", _chunks())

    assert result.ok is False
    assert result.used == []
    assert result.understanding.get("_synthesis_failed") is True
    assert "totally not json" not in json.dumps(result.understanding)  # raw not smuggled in


def test_cache_degrade_keeps_raw_floor() -> None:
    retriever = InMemoryRetriever()
    retriever.add(HANDBOOK, "Leave policy: 21 days of annual leave per year.")
    cache = SemanticCache(retriever, LLMSynthesizer(StubProvider(canned="nope")))

    result = cache.get("leave policy")
    assert result.understanding.get("_synthesis_failed") is True
    assert "21 days" in result.raw_text  # the floor holds even when synthesis fails


def test_custom_instruction_and_fields_flow_into_the_prompt() -> None:
    """You own the content (instruction + fields); Coalent wraps the envelope."""

    class CapturingProvider:
        def __init__(self) -> None:
            self.user = ""

        def generate(self, *, model, system, user, max_tokens, temperature):  # type: ignore[no-untyped-def]
            self.user = user
            return json.dumps({"sla": "99.9%", "used": [0]})

    provider = CapturingProvider()
    synth = LLMSynthesizer(provider, instruction="Extract the SLA only.", fields=["sla"])
    result = synth.synthesize("what is the uptime sla?", _chunks())

    assert "Extract the SLA only." in provider.user   # user instruction is in the prompt
    assert '"sla"' in provider.user                    # custom field requested
    assert result.understanding.get("sla") == "99.9%"  # and returned


def test_citation_fallback_flagged() -> None:
    retriever = InMemoryRetriever()
    retriever.add(HANDBOOK, "Leave policy: 21 days of annual leave per year.")
    canned = json.dumps({"summary": "leave info"})  # valid JSON, but no "used"
    cache = SemanticCache(retriever, LLMSynthesizer(StubProvider(canned=canned)))

    result = cache.get("leave policy")
    assert result.understanding.get("_citation_fallback") is True
