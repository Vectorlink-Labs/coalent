"""Acceptance test — time-based freshness (TTL + revalidate-by-hash).

The feed-less API/tool case: no change webhook, so the cache revalidates on TTL
expiry. Unchanged content stays fresh (no rebuild); changed content rebuilds.
Uses an injected clock for determinism.
"""
from __future__ import annotations

from coalent.semantic import (
    FreshnessPolicy,
    InMemoryRetriever,
    SemanticCache,
    StubSynthesizer,
)

HR = "confluence:hr"


class Clock:
    def __init__(self, start: float = 100.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _cache(policy: FreshnessPolicy, clock: Clock) -> SemanticCache:
    retriever = InMemoryRetriever()
    retriever.add(HR, "leave policy: 21 days of annual leave")
    return SemanticCache(retriever, StubSynthesizer(), freshness=policy, clock=clock)


def test_within_max_age_does_not_revalidate() -> None:
    calls: list[str] = []

    def revalidate(artifact_id: str) -> tuple[str, str]:
        calls.append(artifact_id)
        return ("leave policy: 21 days of annual leave", "v1")

    clock = Clock()
    cache = _cache(FreshnessPolicy(max_age=10, revalidate=revalidate), clock)
    cache.get("leave policy")
    clock.advance(5)  # still within max_age
    result = cache.get("leave policy")

    assert result.cache_hit is True
    assert calls == []  # not revalidated


def test_expiry_unchanged_stays_fresh_no_rebuild() -> None:
    calls: list[str] = []

    def revalidate(artifact_id: str) -> tuple[str, str]:
        calls.append(artifact_id)
        return ("leave policy: 21 days of annual leave", "v1")  # SAME content

    clock = Clock()
    cache = _cache(FreshnessPolicy(max_age=10, revalidate=revalidate), clock)
    first = cache.get("leave policy")
    clock.advance(20)  # expired
    result = cache.get("leave policy")

    assert calls  # it did revalidate
    assert result.cache_hit is True  # unchanged -> stayed fresh, no rebuild
    assert result.unit_id == first.unit_id


def test_expiry_changed_rebuilds() -> None:
    def revalidate(artifact_id: str) -> tuple[str, str]:
        return ("leave policy: now 25 days", "v2")  # CHANGED content

    clock = Clock()
    cache = _cache(FreshnessPolicy(max_age=10, revalidate=revalidate), clock)
    first = cache.get("leave policy")
    clock.advance(20)
    result = cache.get("leave policy")

    assert result.cache_hit is False  # changed -> re-materialized
    assert result.unit_id == first.unit_id


def test_expiry_without_revalidator_rebuilds_conservatively() -> None:
    clock = Clock()
    cache = _cache(FreshnessPolicy(max_age=10), clock)
    cache.get("leave policy")
    clock.advance(20)
    assert cache.get("leave policy").cache_hit is False
