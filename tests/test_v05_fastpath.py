"""v0.5 — fast="auto" equivalence, pinned (the mandatory CI gate for the vectorized twin).

The fast path replaces the three O(cache-size) scans (_best_match / _recall_claims /
_bridge_claims) with matrix math over the SAME control flow. This test runs both paths
side-by-side on the same cache state — dense pseudo-random embeddings, two namespaces,
a dirtied unit, route_by_claim both on and off — and requires identical ids/texts/order
and score agreement to float tolerance. If this fails, fast must not ship."""
from __future__ import annotations

import hashlib
import math
import random

import pytest

from coalent import FunctionEmbedder
from coalent.domain.models import ChangeEvent
from coalent.semantic import Chunk, InMemoryRetriever, SemanticCache, Synthesis
from coalent.semantic.cache import _np

DIM = 32


def _embed(text: str) -> list[float]:
    seed = int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)
    rnd = random.Random(seed)
    v = [rnd.gauss(0.0, 1.0) for _ in range(DIM)]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


class _Synth:
    def synthesize(self, query: str, chunks: list[Chunk]) -> Synthesis:
        text = " ".join(c.text for c in chunks)
        words = text.split()
        half = max(1, len(words) // 2)
        return Synthesis(
            understanding={"summary": text,
                           "claims": [" ".join(words[:half]), " ".join(words[half:])]},
            used=list(range(len(chunks))),
        )


def _mk(route_by_claim: bool) -> SemanticCache:
    retriever = InMemoryRetriever()
    cache = SemanticCache(retriever, _Synth(), embedder=FunctionEmbedder(_embed),
                          hit_threshold=0.95, coverage_floor=0.0,
                          route_by_claim=route_by_claim, read_path="unit")
    topics = ["alpha rates", "beta limits", "gamma policy", "delta pricing",
              "epsilon terms", "zeta quota"]
    for t in topics:
        retriever.add(f"src:{t.split()[0]}", f"{t} document body {t.split()[0]}")
    bar = cache._threshold
    cache._threshold = 2.0
    for t in topics:
        cache.get(f"what about {t}")
    cache._threshold = bar
    cache.invalidate(ChangeEvent(artifact_id="src:zeta"))     # one stale unit in the mix
    return cache


def _both(cache: SemanticCache, fn):                          # run under fast, then pure
    cache._fast_enabled = True
    cache._fast_state = None
    fast = fn()
    cache._fast_enabled = False
    pure = fn()
    return fast, pure


@pytest.mark.skipif(_np is None, reason="numpy not installed — fast path inactive")
@pytest.mark.parametrize("route_by_claim", [False, True])
def test_best_match_equivalent(route_by_claim: bool) -> None:
    cache = _mk(route_by_claim)
    for q in ["alpha rates today", "unrelated cosmic query", "beta limits detail"]:
        qe = tuple(cache._embedder.embed(q))
        (f_id, f_b, f_s), (p_id, p_b, p_s) = _both(cache, lambda: cache._best_match(qe, ""))
        assert f_id == p_id
        assert abs(f_b - p_b) < 1e-9 and abs(f_s - p_s) < 1e-9


@pytest.mark.skipif(_np is None, reason="numpy not installed — fast path inactive")
def test_recall_claims_equivalent_and_masks_stale() -> None:
    cache = _mk(route_by_claim=False)
    qe = tuple(cache._embedder.embed("gamma policy question"))
    fast, pure = _both(cache, lambda: cache._recall_claims(qe, "", limit=7))
    assert [(r.claim, r.unit_id) for r in fast] == [(r.claim, r.unit_id) for r in pure]
    assert all(abs(f.score - p.score) < 1e-9 for f, p in zip(fast, pure))
    stale_ids = {u.id for u in cache._units.values() if not u.is_fresh}
    assert stale_ids and not any(r.unit_id in stale_ids for r in fast)


@pytest.mark.skipif(_np is None, reason="numpy not installed — fast path inactive")
def test_bridge_claims_equivalent() -> None:
    cache = _mk(route_by_claim=False)
    matched = next(u for u in cache._units.values() if u.claim_embeddings and u.is_fresh)
    qe = tuple(cache._embedder.embed("delta pricing"))
    recalled_f, recalled_p = _both(cache, lambda: cache._recall_claims(qe, "", limit=3))
    assert [r.claim for r in recalled_f] == [r.claim for r in recalled_p]
    fast, pure = _both(cache, lambda: cache._bridge_claims(matched, recalled_p, ""))
    assert [(r.claim, r.unit_id) for r in fast] == [(r.claim, r.unit_id) for r in pure]
    assert all(abs(f.score - p.score) < 1e-9 for f, p in zip(fast, pure))


@pytest.mark.skipif(_np is None, reason="numpy not installed — fast path inactive")
def test_end_to_end_get_equivalent() -> None:
    for route in (False, True):
        cache = _mk(route)
        for u in cache._units.values():            # no stale best-match -> no in-place rebuild
            u.mark_fresh()                         # (stale-path equivalence covered above)
        cache._threshold = -1.0                    # force hits: the probe must not mutate state
        for q in ["what about alpha rates", "epsilon terms condition"]:
            cache._fast_enabled = True
            cache._fast_state = None
            n_units = len(cache._units)
            rf = cache.get(q)
            assert len(cache._units) == n_units    # equivalence probe must not build
            cache._fast_enabled = False
            rp = cache.get(q)
            assert len(cache._units) == n_units
            assert rf.cache_hit == rp.cache_hit
            assert rf.unit_id == rp.unit_id
            assert abs(rf.coverage - rp.coverage) < 1e-9
            assert [r.claim for r in rf.recalled] == [r.claim for r in rp.recalled]
