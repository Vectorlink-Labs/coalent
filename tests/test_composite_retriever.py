"""Tests for CompositeRetriever — fan out to many sources, fuse into one unit.

Proves:
  1. evidence from every sub-retriever reaches the cache.
  2. duplicate chunks (same artifact_id + text) collapse, first wins.
  3. one fused cognition unit whose provenance spans ALL sources, so a change to
     any single source invalidates it (the reason to fuse instead of running
     separate caches).
"""
from __future__ import annotations

from coalent.semantic import (
    Chunk,
    CompositeRetriever,
    FunctionRetriever,
    InMemoryRetriever,
    SemanticCache,
    StubSynthesizer,
)


def test_merges_evidence_from_all_retrievers() -> None:
    vec = InMemoryRetriever()
    vec.add("confluence:hr", "Leave policy: 21 days of annual leave.")
    tool = FunctionRetriever(
        lambda q, ns: [Chunk(artifact_id="tool:leave_balance", text='{"balance": 12}')]
    )
    composite = CompositeRetriever([vec, tool])

    ids = {chunk.artifact_id for chunk in composite.retrieve("leave")}
    assert "confluence:hr" in ids
    assert "tool:leave_balance" in ids


def test_collapses_duplicate_chunks_first_wins() -> None:
    dup = Chunk(artifact_id="doc:1", text="same text")
    a = FunctionRetriever(lambda q, ns: [dup])
    b = FunctionRetriever(lambda q, ns: [dup])

    assert len(CompositeRetriever([a, b]).retrieve("x")) == 1


def test_namespace_is_passed_through() -> None:
    vec = InMemoryRetriever()
    vec.add("confluence:hr", "leave policy", namespace="acme")
    vec.add("confluence:other", "unrelated", namespace="other")
    composite = CompositeRetriever([vec])

    ids = {chunk.artifact_id for chunk in composite.retrieve("leave", namespace="acme")}
    assert ids == {"confluence:hr"}


def test_one_fused_unit_invalidated_by_any_source() -> None:
    vec = InMemoryRetriever()
    vec.add("confluence:hr", "Leave policy: 21 days of annual leave.")
    tool = FunctionRetriever(
        lambda q, ns: [Chunk(artifact_id="tool:leave_balance", text="remaining 12 days")]
    )
    cache = SemanticCache(CompositeRetriever([vec, tool]), StubSynthesizer())

    result = cache.get("how much leave do I have")
    assert cache.stats()["units"] == 1                  # ONE fused unit, not two caches
    assert cache.stats()["tracked_artifacts"] == 2      # provenance spans both sources

    # a change to EITHER source dirties the fused unit
    changed = cache.source_changed("tool:leave_balance", text="remaining 5 days")
    assert result.unit_id in changed.dirtied


def test_flaky_sub_retriever_is_skipped_not_fatal() -> None:
    """A throwing source must not blank the fused context or crash the read."""
    healthy = InMemoryRetriever()
    healthy.add("confluence:hr", "Leave policy: 21 days of annual leave.")

    def boom(query: str, namespace: str | None) -> list[Chunk]:
        raise RuntimeError("tool 5xx")

    composite = CompositeRetriever([healthy, FunctionRetriever(boom)])

    # the healthy source's evidence still comes through
    assert {chunk.artifact_id for chunk in composite.retrieve("leave")} == {"confluence:hr"}

    # and a full cache read does not raise — the RAG floor holds
    cache = SemanticCache(composite, StubSynthesizer())
    assert "21 days" in cache.get("leave policy").raw_text
