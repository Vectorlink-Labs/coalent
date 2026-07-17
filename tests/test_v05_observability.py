"""v0.5 — freshness observability (M3), logic tier (hermetic, fake clock).

The freshness guarantee existed since v0.2; v0.5 makes it VISIBLE: an ``on_event`` hook
emitting structured lifecycle events (unit_built / source_changed / stale_read_prevented /
unit_rebuilt) and stats() counters (staleness_prevented, invalidated_units, age-at-serve).
The hard requirements pinned here: events tell the true story of the
build -> serve -> invalidate -> rebuild cycle, ages come from the injected clock, and a
crashing hook can NEVER break a read."""
from __future__ import annotations

from typing import Any

from coalent import StubSynthesizer
from coalent.semantic import InMemoryRetriever, SemanticCache


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def _world() -> tuple[SemanticCache, FakeClock, list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    clock = FakeClock()
    retriever = InMemoryRetriever()
    retriever.add("hr:leave", "Annual leave is 21 days. Carryover max 5 days.")
    cache = SemanticCache(
        retriever, StubSynthesizer(), clock=clock, on_event=events.append,
        hit_threshold=0.1, coverage_floor=0.0,
    )
    return cache, clock, events


def test_lifecycle_events_tell_the_true_story() -> None:
    cache, clock, events = _world()

    r1 = cache.get("annual leave days")                      # cold -> build
    assert r1.cache_hit is False
    assert [e["event"] for e in events] == ["unit_built"]
    assert events[0]["reason"] == "miss" and events[0]["ts"] == clock.now

    clock.now += 60
    r2 = cache.get("annual leave days")                      # warm hit -> no event noise
    assert r2.cache_hit is True
    assert len(events) == 1                                  # serves stay silent by design

    clock.now += 40
    res = cache.source_changed("hr:leave", text="Annual leave is now 25 days.")
    assert res.dirtied
    ev = events[-1]
    assert ev["event"] == "source_changed" and ev["artifact_id"] == "hr:leave"
    assert ev["dirtied"] == res.dirtied and ev["matched_units"] == 1

    clock.now += 20
    r3 = cache.get("annual leave days")                      # stale -> prevented + rebuilt
    assert r3.cache_hit is False
    kinds = [e["event"] for e in events]
    assert kinds == ["unit_built", "source_changed", "stale_read_prevented", "unit_rebuilt"]
    prevented = events[-2]
    assert prevented["unit_id"] == r3.unit_id
    assert prevented["unit_age_s"] == 120.0                  # built at t0, stale-read at t0+120
    assert events[-1]["reason"] == "stale"


def test_stats_counters_and_age_at_serve() -> None:
    cache, clock, _events = _world()
    cache.get("annual leave days")                           # build at t=1000
    clock.now += 100
    cache.get("annual leave days")                           # served at age 100
    clock.now += 200
    cache.get("annual leave days")                           # served at age 300
    s = cache.stats()
    assert s["staleness_prevented"] == 0 and s["invalidated_units"] == 0
    assert s["avg_age_at_serve_s"] == 200.0                  # (100 + 300) / 2
    assert s["max_age_at_serve_s"] == 300.0
    assert s["oldest_fresh_unit_s"] == 300.0

    cache.source_changed("hr:leave", text="changed")
    cache.get("annual leave days")                           # stale read -> prevented + rebuild
    s = cache.stats()
    assert s["staleness_prevented"] == 1
    assert s["invalidated_units"] == 1


def test_no_op_event_for_unmatched_artifact() -> None:
    cache, _clock, events = _world()
    cache.get("annual leave days")
    cache.source_changed("wrong:id", text="whatever")
    ev = events[-1]
    assert ev["event"] == "source_changed" and ev["matched_units"] == 0 and ev["dirtied"] == []
    assert cache.stats()["invalidated_units"] == 0


def test_crashing_hook_never_breaks_a_read() -> None:
    retriever = InMemoryRetriever()
    retriever.add("hr:leave", "Annual leave is 21 days.")

    def bomb(_event: dict[str, Any]) -> None:
        raise RuntimeError("observability outage")

    cache = SemanticCache(retriever, StubSynthesizer(), on_event=bomb,
                          hit_threshold=0.1, coverage_floor=0.0)
    r = cache.get("annual leave days")                       # build emits -> hook raises
    assert r.context                                         # ...and the read still succeeds
    cache.source_changed("hr:leave", text="new")
    r2 = cache.get("annual leave days")                      # stale path emits twice -> fine
    assert r2.cache_hit is False


def test_hook_absent_is_free() -> None:
    retriever = InMemoryRetriever()
    retriever.add("hr:leave", "Annual leave is 21 days.")
    cache = SemanticCache(retriever, StubSynthesizer(), hit_threshold=0.1, coverage_floor=0.0)
    cache.get("annual leave days")
    cache.source_changed("hr:leave", text="new")
    cache.get("annual leave days")
    s = cache.stats()
    assert s["staleness_prevented"] == 1                     # counters work without the hook
