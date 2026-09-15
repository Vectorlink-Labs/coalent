"""v0.7 Phase B increment 1 — Result read-surface readiness (PIPELINE-DESIGN-v07).

The agentic contract (§THE AGENTIC CONTRACT item 4): the Result must ship the read's own
provenance + doubt — read_id, coverage/confidence, needs_retrieval, served sources and
freshness age — so an evaluator node can act without drilling. coverage (final_cov),
confidence (cov0), needs_retrieval and read_id already existed; this increment adds
``sources`` (served artifact ids, served order first, de-duplicated) and
``max_source_age_s`` (MAX served-owner age — the per-read form of the stats() aggregate).

Bar: ZERO serving change — the new fields are computed AFTER pack/render and are never an
input to them (the whole pre-existing suite is the byte-inert pin; the tests here pin the
new fields' presence and content on both read paths).
"""
from __future__ import annotations

import re
import time

from coalent import FunctionEmbedder
from coalent.semantic import Chunk, SemanticCache, Synthesis

_AXES = ("alpha", "beta", "gamma", "kappa", "one", "two", "three", "value", "junk")
_STRIP = re.compile(r"[^\w\s]")


def _embed(text: str) -> list[float]:
    words = set(_STRIP.sub(" ", text.lower()).split())
    v = [1.0 if a in words else 0.0 for a in _AXES]
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v] if n else v


class _ClaimSynth:
    def __init__(self, claims: list[str]) -> None:
        self.claims = list(claims)

    def synthesize(self, query: str, chunks: list[Chunk]) -> Synthesis:
        return Synthesis(understanding={"claims": list(self.claims)},
                         used=list(range(len(chunks))))


class _WordRetriever:
    def __init__(self) -> None:
        self.docs: dict[str, str] = {}

    def set(self, artifact_id: str, text: str) -> None:
        self.docs[artifact_id] = text

    def retrieve(self, query: str, *, namespace: str | None = None) -> list[Chunk]:
        qs = set(_STRIP.sub(" ", query.lower()).split())
        return [Chunk(artifact_id=aid, text=text) for aid, text in self.docs.items()
                if qs & set(_STRIP.sub(" ", text.lower()).split())]


def _header(u) -> str:  # type: ignore[no-untyped-def]
    aids = sorted(u.provenance.artifact_ids())
    return "## " + (aids[0] if aids else u.id)


def _pool(retriever: object, synth: object, **kw: object) -> SemanticCache:
    kw.setdefault("coverage_floor", 0.0)
    kw.setdefault("serve_gate", 0.5)
    kw.setdefault("pool_header", _header)
    kw.setdefault("embedder", FunctionEmbedder(_embed))
    return SemanticCache(retriever, synth,                            # type: ignore[arg-type]
                         read_path="pool", **kw)                      # type: ignore[arg-type]


def _two_unit_world(**kw: object) -> tuple[SemanticCache, str, str]:
    ret = _WordRetriever()
    ret.set("src:a", "alpha value gamma.")
    ret.set("src:b", "beta kappa two.")
    synth = _ClaimSynth(["alpha value"])
    kw.setdefault("serve_budget", 200)
    cache = _pool(ret, synth, **kw)
    ra = cache.get("alpha value")
    synth.claims = ["beta kappa two"]
    rb = cache.get("beta kappa")
    assert ra.unit_id != rb.unit_id
    return cache, ra.unit_id, rb.unit_id


# --------------------------------------------------------------- pool path surface

def test_pool_result_exposes_full_read_surface() -> None:
    # The contract fields, all on one Result: read_id + confidence (cov0) + coverage
    # (final_cov) + needs_retrieval + sources + max_source_age_s.
    cache, a, _b = _two_unit_world()
    r = cache.get("alpha value one")
    assert r.read_id.startswith("read-")
    assert isinstance(r.confidence, float) and isinstance(r.coverage, float)
    assert isinstance(r.needs_retrieval, bool)
    # served order first: the top served owner's artifact leads
    assert r.sources and r.sources[0] == "src:a"
    assert r.pool and r.pool[0].unit_id == a
    # freshness age: freshly built owners serve at ~zero age (wall clock, so a bound)
    assert 0.0 <= r.max_source_age_s < 60.0


def test_pool_sources_deduped_and_age_is_max_served_owner_age() -> None:
    cache, a, b = _two_unit_world()
    # age the FIRST unit only — the read's age must be the MAX across served owners
    cache._units[a].freshness_epoch = time.time() - 500.0
    r = cache.get("alpha value beta kappa")   # budget 200: both owners serve
    served_owners = {c.unit_id for c in r.pool}
    assert served_owners == {a, b}
    assert set(r.sources) == {"src:a", "src:b"}
    assert len(r.sources) == len(set(r.sources))          # de-duplicated
    assert 500.0 <= r.max_source_age_s < 560.0            # max, not min/mean


def test_pool_escalation_raw_artifacts_join_sources() -> None:
    # The RAG floor fires (coverage_floor above every cosine): the escalation raw's
    # artifacts append AFTER the served owners' — including one no unit owns.
    ret = _WordRetriever()
    ret.set("src:a", "alpha value gamma.")
    synth = _ClaimSynth(["alpha value"])
    cache = _pool(ret, synth, serve_budget=200, coverage_floor=0.99)
    cache.get("alpha value")                              # build unit A
    ret.set("src:extra", "alpha extra doc.")              # never built into a unit
    r = cache.get("alpha")                                # serves A, then escalates
    assert r.escalated and r.needs_retrieval
    assert r.sources[0] == "src:a"
    assert "src:extra" in r.sources


# --------------------------------------------------------------- unit path surface

def test_unit_path_result_surface() -> None:
    ret = _WordRetriever()
    ret.set("src:a", "alpha value gamma.")
    cache = SemanticCache(ret, _ClaimSynth(["alpha value"]),          # type: ignore[arg-type]
                          embedder=FunctionEmbedder(_embed),
                          hit_threshold=0.5, coverage_floor=0.0)
    r = cache.get("alpha value gamma")
    assert r.read_id.startswith("read-")
    assert r.sources == ["src:a"]                         # the served evidence's artifacts
    assert 0.0 <= r.max_source_age_s < 60.0
    # a later hit on an aged unit reports the honest age
    cache._units[r.unit_id].freshness_epoch = time.time() - 500.0
    r2 = cache.get("alpha value gamma")
    assert r2.cache_hit
    assert 500.0 <= r2.max_source_age_s < 560.0
