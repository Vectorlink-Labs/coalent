"""Acceptance test for the semantic cache, on the HR scenario.

Proves the four core behaviors:
  1. get(query) works from the query alone.
  2. a rephrased query hits the SAME unit (semantic cache, by meaning).
  3. a detail/number query always has the raw evidence retained.
  4. a source change invalidates exactly that unit and it rebuilds;
     a no-op change (identical content) is skipped.
"""
from __future__ import annotations

from coalent.semantic import InMemoryRetriever, SemanticCache, StubSynthesizer

HANDBOOK = "confluence:hr-handbook"
LEAVE_DOC = (
    "Leave policy: full-time employees accrue 21 days of paid annual leave per year. "
    "Unused leave carries over up to 5 days into the following year."
)
OVERVIEW = "Our HR team supports onboarding, payroll, benefits and leave requests."


def build() -> tuple[SemanticCache, InMemoryRetriever]:
    retriever = InMemoryRetriever()
    retriever.add(HANDBOOK, LEAVE_DOC)
    retriever.add("confluence:hr-overview", OVERVIEW)
    return SemanticCache(retriever, StubSynthesizer()), retriever


def test_get_needs_only_a_query() -> None:
    cache, _ = build()
    result = cache.get("What is our HR leave policy?")  # no domain, no workflow
    assert result.unit_id
    assert result.evidence  # raw retained


def test_rephrase_hits_the_same_unit() -> None:
    cache, _ = build()
    first = cache.get("What is our HR leave policy?")
    assert first.cache_hit is False  # cold build

    second = cache.get("give me details on the HR policy and leaves")
    assert second.cache_hit is True  # semantic hit, different words
    assert second.unit_id == first.unit_id
    assert second.confidence >= 0.6


def test_detail_query_always_has_the_raw_floor() -> None:
    cache, _ = build()
    cache.get("What is our HR leave policy?")

    # A rephrase that needs a specific number the thin understanding never kept.
    result = cache.get("give me details on the HR leave policy")
    assert result.cache_hit is True
    assert "21 days" in result.raw_text                 # the detail is in the RAW
    assert "21" not in str(result.understanding)        # not in the (stub) summary


def test_source_change_invalidates_then_rebuilds() -> None:
    cache, _ = build()
    first = cache.get("What is our HR leave policy?")

    # No-op: identical content must NOT dirty the unit.
    noop = cache.source_changed(HANDBOOK, text=LEAVE_DOC)
    assert first.unit_id in noop.skipped_unchanged
    assert cache.get("HR leave policy details").cache_hit is True

    # Real change: the unit that used this source goes dirty.
    changed = cache.source_changed(HANDBOOK, text=LEAVE_DOC + " Updated 2026: now 25 days.")
    assert first.unit_id in changed.dirtied

    # Next matching read re-materializes the SAME unit (fresh, not a duplicate).
    rebuilt = cache.get("HR leave policy details")
    assert rebuilt.cache_hit is False
    assert rebuilt.unit_id == first.unit_id
    assert cache.stats()["units"] == 1


def test_unrelated_change_touches_nothing() -> None:
    cache, _ = build()
    cache.get("What is our HR leave policy?")
    result = cache.source_changed("confluence:some-other-doc", text="unrelated")
    assert result.dirtied == []
    assert result.skipped_unchanged == []
