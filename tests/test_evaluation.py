"""Acceptance test — the honest cheap+fresh triangle.

Asserts, against an INDEPENDENT oracle and real token cost, that Coalent:
  * is never less correct than naive RAG (the floor),
  * matches naive RAG's freshness (zero stale) — unlike a provenance-less cache,
  * does it at far below naive RAG's cost (only re-materializes what changed).
"""
from __future__ import annotations

from coalent.evaluation import run_benchmark


def test_cheap_and_fresh_triangle() -> None:
    reports = run_benchmark()
    naive = reports["NaiveRAG"]
    stale = reports["StaleCache"]
    coalent = reports["Coalent"]

    # Floor: Coalent is never less correct than naive RAG.
    assert naive.accuracy == 1.0
    assert coalent.accuracy == 1.0

    # Freshness: Coalent matches naive (zero stale); the provenance-less cache goes stale.
    assert naive.stale_rate == 0.0
    assert coalent.stale_rate == 0.0
    assert stale.stale_rate > 0.0
    assert stale.accuracy < 1.0

    # Cost: Coalent is far cheaper than always-fresh, and pays a little more than
    # never-invalidate (the honest price of staying fresh).
    assert coalent.cost_tokens < naive.cost_tokens
    assert coalent.cost_tokens > stale.cost_tokens


def test_reports_cover_every_read() -> None:
    reports = run_benchmark()
    for report in reports.values():
        # 5 topics x 2 phases = 10 reads, each scored as correct or stale-or-miss
        assert report.reads == 10
        assert report.correct + report.stale <= report.reads
