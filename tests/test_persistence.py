"""Acceptance test — durable persistence + invalidation hygiene.

Proves:
  1. units + their invalidation graph survive a "restart" (new store + cache over
     the same DB) — so source-change invalidation still fires (fixes storage-1).
  2. a delete event evicts the units that depended on the source.
  3. a change event matching no cached units is reported (matched_units == 0)
     so the #1 wiring mistake (id mismatch) is catchable.
"""
from __future__ import annotations

from coalent.semantic import (
    InMemoryRetriever,
    SemanticCache,
    SQLiteCognitionStore,
    StubSynthesizer,
)

HR = "confluence:hr-handbook"


def _retriever() -> InMemoryRetriever:
    retriever = InMemoryRetriever()
    retriever.add(HR, "leave policy: 21 days of annual leave per year")
    return retriever


def test_units_and_invalidation_survive_restart(tmp_path) -> None:
    db = str(tmp_path / "cognition.db")
    retriever = _retriever()

    # --- session 1: build a unit, persist it ---
    store1 = SQLiteCognitionStore(db)
    cache1 = SemanticCache(retriever, StubSynthesizer(), store=store1)
    first = cache1.get("what is our leave policy")
    assert first.cache_hit is False
    assert len(store1) == 1
    store1.close()

    # --- session 2 (restart): fresh store + cache over the same DB ---
    store2 = SQLiteCognitionStore(db)
    cache2 = SemanticCache(retriever, StubSynthesizer(), store=store2)

    # the unit persisted -> a semantic hit, not a rebuild
    second = cache2.get("what is our leave policy")
    assert second.cache_hit is True
    assert second.unit_id == first.unit_id

    # invalidation still works after restart (indexes were rebuilt) -> storage-1 fixed
    changed = cache2.source_changed(HR, text="leave policy: now 25 days")
    assert first.unit_id in changed.dirtied
    store2.close()


def test_delete_event_evicts_dependent_units() -> None:
    cache = SemanticCache(_retriever(), StubSynthesizer())
    first = cache.get("leave policy")
    assert cache.stats()["units"] == 1

    result = cache.source_deleted(HR)
    assert first.unit_id in result.deleted
    assert cache.stats()["units"] == 0

    # a later read rebuilds a fresh unit (cold), not a stale hit
    again = cache.get("leave policy")
    assert again.cache_hit is False


def test_zero_match_change_is_reported() -> None:
    cache = SemanticCache(_retriever(), StubSynthesizer())
    cache.get("leave policy")

    result = cache.source_changed("does-not-exist:999", text="whatever")
    assert result.matched_units == 0
    assert result.dirtied == []
    assert result.skipped_unchanged == []
