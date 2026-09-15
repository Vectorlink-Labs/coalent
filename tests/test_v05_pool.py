"""v0.5 — serve="pool" (the experimental preview of the v0.6 pool-first read path), pinned.

Measured basis (held-out n=605, pre-registered): pool serving 0.699@1036 vs the unit-anchored
path 0.579@706 (McNemar z=6.66); ties naive-k9 at 0.79x its tokens. These tests pin the
CONTRACT, not the accuracy: (1) the served payload is the global fresh-claim pool; (2) stale
units' claims are masked the moment a source changes (the freshness covenant); (3) the token
budget caps the payload; (4) the pool_header hook decorates groups and can never break serving;
(5) decision machinery (hit/build) is untouched by the serve swap."""
from __future__ import annotations

from coalent import FunctionEmbedder
from coalent.domain.models import ChangeEvent
from coalent.semantic import Chunk, InMemoryRetriever, SemanticCache, Synthesis

_AXES = ("common", "alpha", "beta", "gamma", "delta", "target", "unique")


def _embed(text: str) -> list[float]:
    words = set(text.lower().split())
    v = [1.0 if a in words else 0.0 for a in _AXES]
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v] if n else v


class _Synth:
    def synthesize(self, query: str, chunks: list[Chunk]) -> Synthesis:
        text = " ".join(c.text for c in chunks)
        return Synthesis(understanding={"summary": text, "claims": [text]},
                         used=list(range(len(chunks))))


def _pool_cache(**kw: object) -> SemanticCache:
    retriever = InMemoryRetriever()
    cache = SemanticCache(retriever, _Synth(), embedder=FunctionEmbedder(_embed),
                          hit_threshold=0.30, coverage_floor=0.0,
                          serve="pool", read_path="unit", **kw)  # type: ignore[arg-type]
    for t in ("alpha", "beta", "gamma"):
        retriever.add(f"src:{t}", f"{t} fact")      # disjoint sources: one topic per artifact
    for t in ("alpha", "beta", "gamma"):
        cache.get(f"common {t}")                    # each seeds exactly one unit
    return cache


def test_pool_context_served_and_spans_units() -> None:
    cache = _pool_cache()
    r = cache.get("common alpha")
    assert r.context.get("serve") == "pool"
    pool = r.context.get("pool", "")
    # the pool payload is global: claims from OTHER units are servable too
    assert "alpha" in pool and ("beta" in pool or "gamma" in pool)


def test_stale_units_claims_are_masked_from_the_pool() -> None:
    cache = _pool_cache()
    alpha_units = [u for u in cache._units.values()
                   if any("alpha" in str(c) for c in u.understanding.get("claims", []))]
    assert alpha_units
    cache.invalidate(ChangeEvent(artifact_id=alpha_units[0].evidence[0].artifact_id))
    r = cache.get("common beta")                    # a read that does NOT rebuild alpha
    pool = r.context.get("pool", "")
    assert "beta" in pool
    assert "alpha" not in pool                      # stale knowledge never serves


def test_serve_budget_caps_the_payload() -> None:
    cache = _pool_cache(serve_budget=8)             # ~8 tokens -> one short claim group
    r = cache.get("common alpha")
    pool = r.context.get("pool", "")
    assert pool
    assert len(pool) // 4 <= 8 + 24                 # budget + one-claim overshoot allowance


def test_pool_header_decorates_and_cannot_break_serving() -> None:
    cache = _pool_cache(pool_header=lambda u: f"[{u.id}]")
    r = cache.get("common alpha")
    assert "[cog:" in r.context.get("pool", "")

    def boom(u: object) -> str:
        raise RuntimeError("hook exploded")

    cache2 = _pool_cache(pool_header=boom)
    r2 = cache2.get("common alpha")
    assert r2.context.get("pool")                   # serving survived the hook


def test_decision_machinery_is_untouched_by_serve_swap() -> None:
    pool_cache = _pool_cache()
    pool_cache._retriever.add("src:unique", "unique fact")  # type: ignore[attr-defined]
    n = len(pool_cache._units)
    miss = pool_cache.get("target unique")          # off-topic -> should still BUILD
    assert miss.cache_hit is False
    assert len(pool_cache._units) == n + 1
    hit = pool_cache.get("common alpha")
    assert hit.cache_hit is True                    # hits still hit


def test_serve_param_validated() -> None:
    retriever = InMemoryRetriever()
    try:
        SemanticCache(retriever, _Synth(), embedder=FunctionEmbedder(_embed), serve="banana")
    except ValueError as e:
        assert "serve" in str(e)
    else:  # pragma: no cover
        raise AssertionError("invalid serve value must raise")


def test_pool_is_namespace_isolated() -> None:              # 0.5.1 fix (D1)
    retriever = InMemoryRetriever()
    cache = SemanticCache(retriever, _Synth(), embedder=FunctionEmbedder(_embed),
                          hit_threshold=0.30, coverage_floor=0.0, serve="pool",
                          read_path="unit")
    retriever.add("src:alpha", "alpha fact")
    retriever.add("src:beta", "beta fact")
    cache.get("common alpha", namespace="team-a")
    cache.get("common beta", namespace="team-b")
    r = cache.get("common alpha", namespace="team-a")
    pool = r.context.get("pool", "")
    assert "alpha" in pool
    assert "beta" not in pool                               # team-b's claims never leak


def test_containment_ns_scoped_both_modes() -> None:        # 0.5.1 fix (D3)
    retriever = InMemoryRetriever()
    cache = SemanticCache(retriever, _Synth(), embedder=FunctionEmbedder(_embed),
                          hit_threshold=0.30, coverage_floor=0.0,
                          provenance_admission=True, read_path="unit")
    retriever.add("src:alpha", "alpha fact")
    cache.get("common alpha", namespace="team-a")           # team-a understands src:alpha
    n = len(cache._units)
    r = cache.get("target alpha", namespace="team-b")       # team-b must BUILD its own
    assert r.cache_hit is False
    assert len(cache._units) == n + 1                       # foreign unit didn't suppress it
