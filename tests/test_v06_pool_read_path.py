"""v0.6 — the pool-first read path (foundation + read path), regression-pinned.

guard tests (V06-REGRESSION-BATTERY.md §3, the -relevant risks):

* ``test_pool_epoch_bumps_on_in_place_rebuild`` / ``test_pool_epoch_monotone_no_cancellation``
  — the CRITICAL epoch-cancellation risk: the deleted ``(len, xor-hash)`` marker could be
  restored by a same-size in-place rebuild, serving OLD claim texts from a memoized pool.
  ``_rows_epoch`` is monotone and cannot cancel.
* ``test_freshness_mask_never_cached_localpool`` — freshness is a LIVE pull-mask, evaluated
  per owner per scan, never snapshotted into an index row.
* ``test_bare_claim_index_second_namespace_raises`` — a bare ClaimIndex used from a second
  namespace raises (the D1 cross-namespace-leak guard), checked on the resolver both add and
  search funnel through.
* ``test_pure_numpy_equivalence`` — the numpy and pure-Python paths return identical ClaimRef
  sequences (scores within the float32 bound) via the shared contract tie-break.
* ``test_usage_summed_multi_build`` — D4: a multi-source read reports the TOTAL synthesis cost.

adds the ``read_path="pool"`` state machine (spec §2.4 P0–P8): the constructor guard,
the P1 reuse channel (stale-rebuild-FIRST), the P2 pool scan (live pull-mask), the P3
bounded TTL, the P4 gate (explicit-absolute / null-shaped adaptive + ceiling), the P5
probe-classify build with caps + structural admission, the P6 S2 band, the P7 RAG floor
(incl. the S2-demoted retrieve-now), and the P8 packing/render contract + rerank hook —
each pinned by the named guard tests below.
"""
from __future__ import annotations

import random

import pytest

from coalent import FunctionEmbedder
from coalent.domain.models import ProvenanceManifest, SourceSpan
from coalent.semantic import (
    Chunk,
    Cognition,
    FreshnessPolicy,
    HashingEmbedder,
    InMemoryRetriever,
    LocalClaimIndex,
    SemanticCache,
    Synthesis,
    Usage,
)
from coalent.semantic import cache as cache_module
from coalent.semantic.cache import (
    _POOL_STAGE1_RAW,
    _POOL_STAGE1_WIDTH,
    _PoolHit,
)
from coalent.semantic.pool import ClaimRef, _np

_AXES = ("common", "alpha", "beta", "gamma", "stale", "revised",
         "value", "mix", "one", "two", "target", "unique",
         "delta", "junk", "p0", "p1", "p2", "p3", "p4", "p5", "other")


def _embed(text: str) -> list[float]:
    words = set(text.lower().split())
    v = [1.0 if a in words else 0.0 for a in _AXES]
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v] if n else v


class _Synth:
    """Echoes chunk text into a single claim — so a rebuild changes the served claim text."""

    def synthesize(self, query: str, chunks: list[Chunk]) -> Synthesis:
        text = " ".join(c.text for c in chunks)
        return Synthesis(understanding={"summary": text, "claims": [text]},
                         used=list(range(len(chunks))))


class _UsageSynth:
    """Reports a fixed per-call token usage so a multi-build read's SUM is checkable (D4)."""

    def __init__(self, prompt: int, completion: int) -> None:
        self._u = Usage(prompt_tokens=prompt, completion_tokens=completion)

    def synthesize(self, query: str, chunks: list[Chunk]) -> Synthesis:
        text = " ".join(c.text for c in chunks)
        return Synthesis(understanding={"summary": text, "claims": [text]},
                         used=list(range(len(chunks))), usage=self._u)


class _MutableRetriever:
    """A retriever whose source texts can be edited in place — to drive an in-place rebuild."""

    def __init__(self) -> None:
        self.docs: dict[str, str] = {}

    def set(self, artifact_id: str, text: str) -> None:
        self.docs[artifact_id] = text

    def retrieve(self, query: str, *, namespace: str | None = None) -> list[Chunk]:
        qs = set(query.lower().split())
        return [Chunk(artifact_id=aid, text=text)
                for aid, text in self.docs.items()
                if qs & set(text.lower().split())]


def _mk_cache(**kw: object) -> SemanticCache:
    return SemanticCache(InMemoryRetriever(), _Synth(), embedder=FunctionEmbedder(_embed),
                         hit_threshold=0.99, coverage_floor=0.0, **kw)  # type: ignore[arg-type]


# ----------------------------------------------------------------- LocalClaimIndex freshness

def test_freshness_mask_never_cached_localpool() -> None:
    fresh = {"u1": True, "u2": True}
    idx = LocalClaimIndex(fresh_of=lambda uid: fresh[uid])
    idx.add("u1", ["alpha one", "alpha two"], [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]])
    idx.add("u2", ["beta one"], [[0.0, 1.0, 0.0]])
    q = [1.0, 0.0, 0.0]

    owners = {ref.unit_id for _, ref, _ in idx.search(q, 10)}
    assert owners == {"u1", "u2"}

    # Flip u1 stale WITHOUT re-adding: a live pull-mask removes its rows on the very next scan.
    fresh["u1"] = False
    res2 = idx.search(q, 10)
    assert {ref.unit_id for _, ref, _ in res2} == {"u2"}
    assert all(ref.unit_id != "u1" for _, ref, _ in res2)

    # Flip back fresh -> reappears (freshness is never snapshotted in either direction).
    fresh["u1"] = True
    assert {ref.unit_id for _, ref, _ in idx.search(q, 10)} == {"u1", "u2"}

    # include_stale is telemetry-only: masked rows surface flagged fresh=False.
    fresh["u1"] = False
    stale = [(ref.unit_id, is_fresh) for _, ref, is_fresh in idx.search(q, 10, include_stale=True)]
    assert ("u1", False) in stale
    assert all(f for uid, f in stale if uid == "u2")


def test_localpool_add_replaces_and_remove_drops() -> None:
    idx = LocalClaimIndex()
    assert idx.add("u1", ["a", "b"], [[1.0, 0.0], [0.0, 1.0]]) == 2
    assert len(idx) == 2
    # add is REPLACE semantics, not append
    assert idx.add("u1", ["c"], [[1.0, 1.0]]) == 1
    assert len(idx) == 1
    # blank text / zero embeddings are skipped, never indexed
    assert idx.add("u2", ["", "real"], [[1.0, 0.0], [0.0, 0.0]]) == 0
    assert len(idx) == 1
    idx.remove("u1")
    assert len(idx) == 0
    idx.remove("does-not-exist")   # never raises on unknown ids


def test_localpool_dim_and_length_guards() -> None:
    idx = LocalClaimIndex()
    idx.add("u1", ["a"], [[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="dim"):
        idx.add("u2", ["b"], [[1.0, 0.0]])            # dimension changed (embedder swap)
    with pytest.raises(ValueError, match="length"):
        idx.add("u3", ["b", "c"], [[1.0, 0.0, 0.0]])  # claims/embs mismatch


# ------------------------------------------------------------------- numpy/pure equivalence

@pytest.mark.skipif(_np is None, reason="numpy not installed — numpy path inactive")
def test_pure_numpy_equivalence() -> None:
    rnd = random.Random(1234)

    def vec() -> list[float]:
        return [rnd.gauss(0.0, 1.0) for _ in range(16)]

    numpy_idx = LocalClaimIndex(use_numpy=True)
    pure_idx = LocalClaimIndex(use_numpy=False)
    assert numpy_idx._use_numpy is True and pure_idx._use_numpy is False

    for i in range(12):
        uid = f"u{i}"
        claims = [f"{uid} claim {j}" for j in range(rnd.randint(1, 4))]
        embs = [vec() for _ in claims]
        numpy_idx.add(uid, claims, embs)
        pure_idx.add(uid, claims, embs)

    for _ in range(20):
        q = vec()
        rn = numpy_idx.search(q, 25)
        rp = pure_idx.search(q, 25)
        assert ([(r.unit_id, r.claim_idx, r.text) for _, r, _ in rn]
                == [(r.unit_id, r.claim_idx, r.text) for _, r, _ in rp])
        for (sn, _, _), (sp, _, _) in zip(rn, rp):
            assert abs(sn - sp) < 1e-6

    # Exact-tie determinism: identical embeddings resolve by (unit_id, claim_idx) in BOTH paths.
    tie = [1.0] + [0.0] * 15
    for ix in (numpy_idx, pure_idx):
        ix.add("zzz", ["z"], [tie])
        ix.add("aaa", ["a"], [tie])
    order_n = [r.unit_id for _, r, _ in numpy_idx.search(tie, 50)]
    order_p = [r.unit_id for _, r, _ in pure_idx.search(tie, 50)]
    assert order_n == order_p
    assert order_n.index("aaa") < order_n.index("zzz")   # tie-break: smaller unit_id first


def test_localpool_search_respects_top_n_and_tie_break_pure() -> None:
    idx = LocalClaimIndex(use_numpy=False)
    tie = [1.0, 0.0]
    idx.add("beta", ["x"], [tie])
    idx.add("alpha", ["y", "z"], [tie, tie])
    res = idx.search([1.0, 0.0], 2)
    assert len(res) == 2
    # (-score, unit_id, claim_idx): all score 1.0 -> alpha#0, alpha#1, beta#0
    assert [(r.unit_id, r.claim_idx) for _, r, _ in res] == [("alpha", 0), ("alpha", 1)]


# --------------------------------------------------------------------- namespace guard (D1)

def test_bare_claim_index_second_namespace_raises() -> None:
    shared = LocalClaimIndex()
    cache = _mk_cache(claim_index=shared)

    # First namespace binds the bare instance (the resolution the build/add path performs).
    assert cache._resolve_claim_index("team-a") is shared
    # Same namespace re-resolves to the same instance — no error.
    assert cache._resolve_claim_index("team-a") is shared
    # A SECOND namespace (the resolution the read/search path performs) must raise + name the fix.
    with pytest.raises(ValueError, match="factory"):
        cache._resolve_claim_index("team-b")

    # A factory mints one index per namespace with no error (the prescribed remedy).
    cache2 = _mk_cache(claim_index=lambda ns: LocalClaimIndex())
    a = cache2._resolve_claim_index("team-a")
    b = cache2._resolve_claim_index("team-b")
    assert a is not b
    assert cache2._resolve_claim_index("team-a") is a

    # Default (None) also mints per-namespace built-ins.
    cache3 = _mk_cache()
    idx = cache3._resolve_claim_index("x")
    assert isinstance(idx, LocalClaimIndex)
    assert cache3._resolve_claim_index("x") is idx


# --------------------------------------------------------------------- rows_epoch monotonicity

def test_pool_epoch_bumps_on_in_place_rebuild() -> None:
    ret = _MutableRetriever()
    ret.set("src:a", "alpha stale value")
    cache = SemanticCache(ret, _Synth(), embedder=FunctionEmbedder(_embed),
                          hit_threshold=0.3, coverage_floor=0.0, serve="pool",
                          read_path="unit")

    r0 = cache.get("alpha")
    assert r0.cache_hit is False
    e0 = cache._rows_epoch
    assert e0 > 0
    assert "stale" in r0.context.get("pool", "")

    # A source change DIRTIES the unit: not a row mutation (epoch unchanged), status flips.
    g0 = cache._status_gen
    ret.set("src:a", "alpha revised value")
    cache.source_changed("src:a", text="alpha revised value")
    assert cache._rows_epoch == e0
    assert cache._status_gen > g0

    # Re-query: the stale unit rebuilds IN PLACE (same id) — rows_epoch BUMPS, never cancels.
    r1 = cache.get("alpha")
    assert r1.cache_hit is False           # a rebuild ran
    assert r1.unit_id == r0.unit_id        # same unit id, rebuilt in place (no duplicate)
    assert cache._rows_epoch > e0

    # The memoized pool must now serve the REBUILT claim, never the stale text (the marker bug).
    pool = r1.context.get("pool", "")
    assert "revised" in pool
    assert "stale" not in pool


def test_pool_epoch_monotone_no_cancellation() -> None:
    ret = InMemoryRetriever()
    ret.add("src:a", "alpha value")
    ret.add("src:b", "beta value")
    cache = SemanticCache(ret, _Synth(), embedder=FunctionEmbedder(_embed),
                          hit_threshold=0.99, coverage_floor=0.0, serve="pool",
                          read_path="unit")

    cache.get("alpha")
    e_a = cache._rows_epoch
    cache.get("beta")
    e_b = cache._rows_epoch
    assert e_b > e_a                       # building B advanced the epoch

    # Evict B: membership returns to {A only} — a (len, xor-hash) marker WOULD collide with the
    # earlier {A}-only state. The monotone counter cannot cancel back to a prior value.
    b_units = [u for u in cache._units.values()
               if any("beta" in str(c) for c in u.understanding.get("claims", []))]
    assert b_units
    cache.source_deleted(b_units[0].evidence[0].artifact_id)
    e_c = cache._rows_epoch
    assert e_c > e_b
    assert e_c != e_a                      # never restored to an earlier marker value

    # And every mutation strictly advanced it — a monotone, cancellation-free sequence.
    assert 0 < e_a < e_b < e_c


# ------------------------------------------------------------------------ D4 usage summation

def test_usage_summed_multi_build() -> None:
    ret = InMemoryRetriever()
    ret.add("src:one", "mix alpha one")
    ret.add("src:two", "mix beta two")
    cache = SemanticCache(ret, _UsageSynth(prompt=10, completion=5),
                          embedder=FunctionEmbedder(_embed),
                          hit_threshold=0.99, coverage_floor=0.0, split_by_artifact=True,
                          read_path="unit")

    r = cache.get("common mix")
    assert r.cache_hit is False
    # Two artifacts -> two synthesis calls -> Result.usage is the SUM across ALL builds (D4).
    assert r.usage is not None
    assert r.usage.prompt_tokens == 20
    assert r.usage.completion_tokens == 10
    assert r.usage.total_tokens == 30
    # stats totals corroborate (they were already correct; the D4 fix is the per-read Result.usage).
    assert cache.stats()["synth_tokens"] == 30
    assert cache.stats()["synth_calls"] == 2


# ═══════════════════════════════════════════ — the pool-first read path (read_path="pool")

class _EchoSynth:
    """Echoes each build's chunk text into one claim (+ summary) and counts calls —
    the pool tests' synthesis-op meter (stats()['synth_calls'] only counts usage-carrying
    calls, so a plain counter is the honest measure of 'did we pay an LLM')."""

    def __init__(self) -> None:
        self.calls = 0

    def synthesize(self, query: str, chunks: list[Chunk]) -> Synthesis:
        self.calls += 1
        text = " ".join(c.text for c in chunks)
        return Synthesis(understanding={"summary": text, "claims": [text]},
                         used=list(range(len(chunks))))


class _FailSynth:
    """Synthesis always fails (ok=False) — the _synthesis_failed containment guard."""

    def __init__(self) -> None:
        self.calls = 0

    def synthesize(self, query: str, chunks: list[Chunk]) -> Synthesis:
        self.calls += 1
        return Synthesis(understanding={}, ok=False)


class _CountingRetriever(InMemoryRetriever):
    """InMemoryRetriever that counts query-shaped retrieve() calls (the <=1/read invariant)."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def retrieve(self, query: str, *, namespace: str | None = None) -> list[Chunk]:
        self.calls += 1
        return super().retrieve(query, namespace=namespace)


class _CountingMutableRetriever(_MutableRetriever):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def retrieve(self, query: str, *, namespace: str | None = None) -> list[Chunk]:
        self.calls += 1
        return super().retrieve(query, namespace=namespace)


def _pool(retriever: object = None, synth: object = None, **kw: object) -> SemanticCache:
    kw.setdefault("coverage_floor", 0.0)   # escalation floor OFF unless a test arms it
    return SemanticCache(
        retriever if retriever is not None else InMemoryRetriever(),  # type: ignore[arg-type]
        synth if synth is not None else _EchoSynth(),                 # type: ignore[arg-type]
        embedder=FunctionEmbedder(_embed), read_path="pool", **kw)    # type: ignore[arg-type]


def _mk_unit(uid: str, ns: str, qtext: str, claims: list[str], artifact: str) -> Cognition:
    emb = tuple(_embed(qtext))
    return Cognition(
        id=uid, namespace=ns, query=qtext, query_embedding=emb,
        understanding={"claims": list(claims)}, evidence=(),
        provenance=ProvenanceManifest(
            "synth@1", "semantic@2",
            source_spans=(SourceSpan.from_text(artifact, "doc"),)),
        understanding_embedding=emb,
        claim_embeddings=tuple(tuple(_embed(c)) for c in claims),
    )


# ------------------------------------------------------------ constructor guards (spec §2.1)

def test_pool_requires_semantic_embedder() -> None:
    # Constructor-time, and the error text names the escape hatches.
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        SemanticCache(InMemoryRetriever(), _EchoSynth(),          # type: ignore[arg-type]
                      embedder=HashingEmbedder(), read_path="pool")
    with pytest.raises(ValueError, match="embedder="):
        SemanticCache(InMemoryRetriever(), _EchoSynth(),          # type: ignore[arg-type]
                      read_path="pool")   # conftest defaults to HashingEmbedder
    # The unit path keeps constructing under HashingEmbedder (v0.5 unchanged).
    SemanticCache(InMemoryRetriever(), _EchoSynth())              # type: ignore[arg-type]
    with pytest.raises(ValueError, match="read_path"):
        SemanticCache(InMemoryRetriever(), _EchoSynth(),          # type: ignore[arg-type]
                      embedder=FunctionEmbedder(_embed), read_path="both")


def test_serve_budget_default_split() -> None:
    ret = InMemoryRetriever()
    assert SemanticCache(ret, _EchoSynth())._serve_budget == 600  # type: ignore[arg-type]
    assert _pool()._serve_budget == 1000
    assert _pool(serve_budget=250)._serve_budget == 250
    assert SemanticCache(ret, _EchoSynth(),                        # type: ignore[arg-type]
                         serve_budget=250)._serve_budget == 250
    with pytest.raises(ValueError, match="serve_budget"):
        _pool(serve_budget=0)
    with pytest.raises(ValueError, match="serve_budget"):
        SemanticCache(ret, _EchoSynth(), serve_budget=-5)          # type: ignore[arg-type]


# ------------------------------------------------------------------ read path & gates (P0-P5)

def test_pool_serve_when_covered() -> None:
    ret = _CountingRetriever()
    ret.add("src:a", "alpha value one")
    cache = _pool(ret, serve_gate=0.5)
    r1 = cache.get("alpha value one")
    assert r1.cache_hit is False
    n = ret.calls
    r2 = cache.get("alpha one")          # cos ~0.816 to the claim; < 0.9 to the seed
    assert r2.cache_hit is True
    assert ret.calls == n                # a gate-pass serve retrieves NOTHING
    assert r2.understanding == {"claims": ["alpha value one"]}
    assert r2.context["serve"] == "pool"
    assert r2.evidence == []             # citations via drill(unit_id)


def test_pool_build_on_low_coverage() -> None:
    ret = InMemoryRetriever()
    ret.add("src:a", "alpha value one")
    ret.add("src:g", "gamma two")
    cache = _pool(ret, serve_gate=0.5)
    cache.get("alpha value one")
    r = cache.get("gamma two")           # no fresh claim covers it -> gap build
    assert r.cache_hit is False
    assert "gamma two" in r.understanding["claims"]


def test_empty_pool_builds_even_at_zero_gate() -> None:
    """serve_gate=0.0 (explicit-absolute) on a COLD pool: zero candidates is never a serve.

    Regression: the P4 comparison cov0 >= gate passed as 0.0 >= 0.0 on an empty pool, so an
    operator-pinned zero gate served empty forever and the gap build could never trigger. The
    gate arbitrates among candidates; with no fresh rows the read must probe-build.
    """
    events: list[dict] = []
    ret = InMemoryRetriever()
    ret.add("src:a", "alpha value one")
    cache = _pool(ret, serve_gate=0.0, on_event=events.append)

    r1 = cache.get("alpha value one")
    assert r1.cache_hit is False                       # first read BUILDS...
    assert "alpha value one" in r1.understanding["claims"]   # ...and serves content, not empty
    gates = [e for e in events if e["event"] == "pool_gate"]
    assert gates[0]["outcome"] == "build"

    r2 = cache.get("alpha one")                        # now there ARE candidates: 0.0 gate serves
    assert r2.cache_hit is True
    assert r2.understanding == {"claims": ["alpha value one"]}
    assert [e["outcome"] for e in events if e["event"] == "pool_gate"] == ["build", "serve"]


def test_reuse_channel_bypasses_gate() -> None:
    events: list[dict] = []
    ret = InMemoryRetriever()
    ret.add("src:a", "alpha value one")
    cache = _pool(ret, serve_gate=0.99, on_event=events.append)   # nothing clears the gate
    synth = cache._synth
    cache.get("alpha value one")                                  # build (below gate)
    r2 = cache.get("alpha value one")                             # identical -> forced serve
    assert r2.cache_hit is True
    assert synth.calls == 1                                       # type: ignore[attr-defined]
    gates = [e for e in events if e["event"] == "pool_gate"]
    assert gates[-1]["outcome"] == "reuse"


def test_seed_reuse_stale_rebuilds_before_serve() -> None:
    # Regression-suite name: test_reuse_stale_rebuild_first. A stale P1 reuse match rebuilds
    # IN PLACE before serving (consuming the read's single retrieval) — never a stale serve.
    events: list[dict] = []
    ret = _CountingMutableRetriever()
    ret.set("src:a", "alpha stale value")
    cache = _pool(ret, serve_gate=0.5, on_event=events.append)
    r1 = cache.get("alpha value")
    assert r1.cache_hit is False
    ret.set("src:a", "alpha revised value")
    cache.source_changed("src:a", text="alpha revised value")
    n = ret.calls
    r2 = cache.get("alpha value")        # same question again, but the unit is STALE
    assert r2.cache_hit is False         # a rebuild ran (zero-synthesis definition)
    assert r2.unit_id == r1.unit_id      # same unit, rebuilt in place
    assert ret.calls == n + 1            # the rebuild consumed the ONE retrieval
    pool_text = r2.context["pool"]
    assert "revised" in pool_text and "stale" not in pool_text
    kinds = [e["event"] for e in events]
    assert "stale_read_prevented" in kinds
    rebuilds = [e for e in events if e["event"] == "rebuild_triggered_by_read"]
    assert rebuilds and rebuilds[-1]["reason"] == "reuse_stale"
    assert [e for e in events if e["event"] == "pool_gate"][-1]["outcome"] == "reuse"


def test_single_retrieval_per_read() -> None:
    ret = _CountingRetriever()
    ret.add("src:a", "alpha value one")
    ret.add("src:b", "beta value two")
    cache = _pool(ret, serve_gate=0.5)
    cache.get("alpha value one")          # build read: exactly 1 (probe feeds ALL builds)
    assert ret.calls == 1
    cache.get("alpha one")                # serve read: 0
    assert ret.calls == 1
    cache.get("gamma beta")               # below gate: probe -> classification: exactly 1
    assert ret.calls == 2


def test_admission_reuse_wart_fix() -> None:
    events: list[dict] = []
    ret = _CountingRetriever()
    ret.add("src:a", "target alpha value")
    cache = _pool(ret, serve_gate=0.9, on_event=events.append)
    synth = cache._synth
    cache.get("target alpha value")       # build
    r2 = cache.get("target")              # cov ~0.577 < gate, but the probe is all-CONTAINED
    assert r2.cache_hit is True           # provenance PROVES coverage: zero synthesis
    assert synth.calls == 1               # type: ignore[attr-defined]
    admissions = [e for e in events if e["event"] == "admission_reuse"]
    assert admissions and admissions[0]["probed_sources"] == 1
    s = cache.stats()
    assert s["probe_reads"] == 1 and s["admission_reuses"] == 1


def test_cold_start_empty_pool() -> None:
    cache = _pool(_CountingRetriever())
    r = cache.get("alpha value")
    assert r.cache_hit is True            # zero synthesis ran
    assert r.needs_retrieval is True
    assert r.pool == [] and r.unit_id == "" and r.evidence == []
    assert r.context["pool"] == ""
    assert cache.stats()["probe_reads"] == 1


def test_warm_null_no_synthesis_after_first() -> None:
    ret = InMemoryRetriever()
    ret.add("src:junk", "junk gamma")
    cache = _pool(ret, serve_gate=0.9)
    synth = cache._synth
    cache.get("gamma one")                # first null: one build (the junk family unit)
    cache.get("gamma two")                # warm null: probe -> contained -> NO synthesis
    cache.get("gamma value")              # and again
    assert synth.calls == 1               # type: ignore[attr-defined]


def test_failed_unit_containment_bounded_retry() -> None:
    ret = _CountingMutableRetriever()
    ret.set("src:a", "alpha value")
    synth = _FailSynth()
    cache = _pool(ret, synth, serve_gate=0.5)
    r1 = cache.get("alpha value")
    assert r1.cache_hit is False
    assert synth.calls == 1
    # A _synthesis_failed unit COUNTS as containing: the next probe must NOT re-pay synthesis.
    r2 = cache.get("alpha one")
    assert r2.cache_hit is True
    assert synth.calls == 1               # bounded retry: no retry-every-read paid loop
    # The retry trigger is provenance: a real content change re-materializes.
    ret.set("src:a", "alpha revised value")
    cache.source_changed("src:a", text="alpha revised value")
    cache.get("alpha two")
    assert synth.calls == 2


def test_builds_capped_at_three_per_read() -> None:
    events: list[dict] = []
    ret = InMemoryRetriever()
    for i in range(5):
        ret.add(f"src:{i}", f"common p{i}")
    cache = _pool(ret, serve_gate=0.99, on_event=events.append)
    synth = cache._synth
    cache.get("common one")
    assert synth.calls == 3               # type: ignore[attr-defined]
    gap = [e for e in events if e["event"] == "build_triggered_by_gap"][-1]
    assert gap["n_units_built"] == 3 and len(gap["artifacts"]) == 3
    cache.get("common two")               # multi-read convergence: the rest build next read
    assert synth.calls == 5               # type: ignore[attr-defined]


def test_usage_summed_across_builds_pool() -> None:
    ret = InMemoryRetriever()
    ret.add("src:one", "mix alpha one")
    ret.add("src:two", "mix beta two")
    cache = SemanticCache(ret, _UsageSynth(prompt=10, completion=5),
                          embedder=FunctionEmbedder(_embed),
                          read_path="pool", serve_gate=0.99, coverage_floor=0.0)
    r = cache.get("common mix")
    assert r.cache_hit is False
    assert r.usage is not None and r.usage.total_tokens == 30    # two builds, SUMMED (D4)


def test_source_anchored_unit_identity() -> None:
    import hashlib as _hl
    ret = InMemoryRetriever()
    ret.add("src:a", "alpha value")
    cache = _pool(ret, serve_gate=0.99)
    cache.get("alpha value")
    expected = "cog:" + _hl.sha1(b"|artifact:src:a").hexdigest()[:16]
    assert expected in cache._units
    cache.get("alpha unique")             # different query, same source: NO duplicate unit
    assert len(cache._units) == 1         # one unit per (namespace, artifact), always


def test_probe_classify_artifact_index_ns_scoped() -> None:
    # D3 guard: a foreign namespace's containing unit must never absorb this namespace's read.
    ret = InMemoryRetriever()
    ret.add("src:x", "target value")
    cache = _pool(ret, serve_gate=0.99, claim_index=lambda ns: LocalClaimIndex())
    r1 = cache.get("target value", namespace="team-a")
    r2 = cache.get("target value", namespace="team-b")
    assert r2.cache_hit is False          # ns-B built its OWN unit — no cross-ns admission
    assert r1.unit_id != r2.unit_id
    ids = {u.namespace for u in cache._units.values()}
    assert ids == {"team-a", "team-b"}


# ----------------------------------------------------------------------- the gate (P4, §2.3)

def test_pool_gate_event_emitted() -> None:
    events: list[dict] = []
    ret = InMemoryRetriever()
    ret.add("src:a", "alpha value one")
    cache = _pool(ret, serve_gate=0.5, on_event=events.append)
    cache.get("alpha value one")
    cache.get("alpha one")
    cache.get("alpha value one")
    gates = [e for e in events if e["event"] == "pool_gate"]
    assert len(gates) == 3                # exactly one per read
    assert [g["outcome"] for g in gates] == ["build", "serve", "reuse"]
    assert all(set(g) >= {"coverage", "gate", "outcome"} for g in gates)


def test_gate_explicit_is_absolute() -> None:
    ret = InMemoryRetriever()
    ret.add("src:a", "alpha value one")
    cache = _pool(ret, serve_gate=0.05)
    cache.get("alpha value one")
    cache._pool_noise_ceiling = 0.99      # adaptation must be DISABLED entirely
    assert cache._effective_pool_gate() == 0.05
    r = cache.get("alpha two")            # cos ~0.408 >= 0.05 -> serves
    assert r.cache_hit is True
    assert cache.stats()["serve_gate_effective"] == 0.05


def test_adaptive_gate_ceiling() -> None:
    # cov_default for a custom embedder is 0.4; the adaptive gate can never exceed 0.4+0.27.
    ret = InMemoryRetriever()
    ret.add("src:a", "alpha")
    cache = _pool(ret)                    # serve_gate=None -> adaptive
    cache.get("alpha")
    cache._pool_noise_ceiling = 0.95      # a dense near-duplicate pool
    assert cache._effective_pool_gate() == pytest.approx(0.67)
    r = cache.get("alpha one")            # a real answer at cos ~0.707 still SERVES
    assert r.cache_hit is True


def test_adaptive_gate_null_shaped_probes() -> None:
    # The probe statistic is PROVENANCE-DISJOINT: a same-artifact twin's cos-1.0 claim is
    # SIGNAL (excluded from noise); only artifact-disjoint same-ns rows count.
    cache = _pool()
    for i in range(6):
        for suffix in ("x", "y"):        # 6 pairs; twins share an artifact + vectors
            u = _mk_unit(f"cog:pair{i}{suffix}", "", f"p{i}", [f"p{i}"], f"art:{i}")
            cache._units[u.id] = u
    # A same-vector unit in ANOTHER namespace is invisible to these probes (ns isolation).
    foreign = _mk_unit("cog:foreign", "other", "p0", ["p0"], "art:foreign")
    cache._units[foreign.id] = foreign
    assert cache._pool_noise_floor() == 0.0   # naive (non-disjoint) sampling would say 1.0
    # Fewer than 8 units -> ceiling 0.0 by contract.
    small = _pool()
    for i in range(3):
        u = _mk_unit(f"cog:s{i}", "", f"p{i}", [f"p{i}"], f"art:{i}")
        small._units[u.id] = u
    assert small._pool_noise_floor() == 0.0


def test_adaptive_gate_recalibrates_at_build_read_end() -> None:
    ret = InMemoryRetriever()
    ret.add("src:t", "target unique")
    cache = _pool(ret)                    # adaptive (serve_gate=None)
    # 12 fresh units whose cross-unit (provenance-disjoint) claim cosine is exactly 0.5.
    for i in range(6):
        for suffix in ("x", "y"):
            uid = f"cog:n{i}{suffix}"
            # Twins share an artifact -> provenance-disjoint probes exclude the cos-1.0 twin;
            # what remains is the 0.5 cross-pair similarity (the honest noise level).
            u = _mk_unit(uid, "", f"p{i} common", [f"p{i} common"], f"art:{i}")
            cache._units[uid] = u
    assert cache._effective_pool_gate() == pytest.approx(0.4)   # pre-calibration: cov_default
    cache.get("target unique")            # a BUILD read: recalibration runs at its END
    assert cache._pool_noise_ceiling == pytest.approx(0.5)
    assert cache._builds_since_calib == 0
    # effective gate = min(max(cov_default, 0.5 + 0.02), cov_default + 0.27) = 0.52
    assert cache._effective_pool_gate() == pytest.approx(0.52)


# ------------------------------------------------------------------------- freshness (P2/P3)

def test_pool_masks_stale_unit_claims_immediately() -> None:
    events: list[dict] = []
    ret = InMemoryRetriever()
    ret.add("src:a", "alpha value")
    ret.add("src:b", "beta value")
    cache = _pool(ret, serve_gate=0.4, on_event=events.append)
    cache.get("alpha value")              # builds both (shared token "value")
    stale_ids = {u.id for u in cache._units.values()
                 if "alpha value" in u.understanding["claims"]}
    cache.source_changed("src:a", text="alpha changed now")
    r = cache.get("alpha beta value")     # serve: B covers; A is stale, invisible NOW
    assert r.cache_hit is True
    assert "alpha value" not in r.understanding["claims"]
    assert all(p.unit_id not in stale_ids for p in r.pool)
    masked = [e for e in events if e["event"] == "pool_masked_stale"]
    assert masked and set(masked[-1]) >= {"n_rows", "units", "top_stale_score"}
    assert set(masked[-1]["units"]) <= stale_ids


def test_freshness_mask_never_cached_pool_read_path() -> None:
    # The LocalClaimIndex property, extended THROUGH the read path: flipping status
    # (no re-add, no rebuild) changes what the very next read serves, both directions.
    ret = InMemoryRetriever()
    ret.add("src:a", "alpha value")
    ret.add("src:b", "beta value")
    cache = _pool(ret, serve_gate=0.4)
    cache.get("alpha value")
    unit_a = next(u for u in cache._units.values()
                  if "alpha value" in u.understanding["claims"])
    epoch = cache._rows_epoch
    cache._dirty(unit_a)
    r1 = cache.get("alpha beta value")
    assert "alpha value" not in r1.understanding["claims"]
    cache._mark_fresh(unit_a)
    r2 = cache.get("alpha beta value")
    assert "alpha value" in r2.understanding["claims"]
    assert cache._rows_epoch == epoch     # pure pull-mask: rows were NEVER touched


def test_stale_rebuild_in_place() -> None:
    events: list[dict] = []
    ret = _CountingMutableRetriever()
    ret.set("src:a", "alpha stale value")
    cache = _pool(ret, serve_gate=0.5, on_event=events.append)
    r1 = cache.get("alpha value")
    ret.set("src:a", "alpha revised value")
    cache.source_changed("src:a", text="alpha revised value")
    r2 = cache.get("alpha unique")        # NOT a reuse (cos to seed ~0.67): the gap path
    assert r2.cache_hit is False
    assert r2.unit_id == r1.unit_id       # rebuilt IN PLACE, no duplicate
    rebuilds = [e for e in events if e["event"] == "rebuild_triggered_by_read"]
    assert rebuilds and rebuilds[-1]["reason"] == "stale"
    assert rebuilds[-1]["unit_id"] == r1.unit_id
    assert "revised" in r2.context["pool"]
    assert cache.stats()["staleness_prevented"] >= 1


def test_delete_event_removes_pool_rows() -> None:
    ret = InMemoryRetriever()
    ret.add("src:a", "alpha value")
    ret.add("src:b", "beta value")
    cache = _pool(ret, serve_gate=0.4)
    cache.get("alpha value")
    idx = cache._claim_indexes[""]
    assert len(idx) == 2
    cache.source_deleted("src:a")
    assert len(idx) == 1                  # rows removed, not merely masked
    r = cache.get("alpha beta value")
    assert r.understanding["claims"] == ["beta value"]


def test_ttl_checks_only_candidate_owners() -> None:
    # Regression-suite name: test_ttl_checks_only_candidate_head. Owners OUTSIDE the candidate
    # head (2x serve_budget of ranked rows) never cost a revalidator fetch — bounded work.
    calls: list[str] = []
    t = [0.0]
    long_text = "target value " * 12

    def reval(artifact_id: str) -> tuple[str, str]:
        calls.append(artifact_id)
        return (long_text if artifact_id == "src:a" else "junk gamma"), "v1"

    ret = InMemoryRetriever()
    ret.add("src:a", long_text)
    ret.add("src:b", "junk gamma")
    cache = _pool(ret, serve_gate=0.3, serve_budget=4, clock=lambda: t[0],
                  freshness=FreshnessPolicy(max_age=10, revalidate=reval))
    cache.get("target value")
    cache.get("junk gamma")
    t[0] = 100.0                          # both owners are now past max_age
    r = cache.get("target one")           # only src:a's owner is in the candidate head
    assert r.cache_hit is True
    assert calls == ["src:a"]             # src:b expired but NEVER checked this read


def test_ttl_revalidation_capped_per_read() -> None:
    calls: list[str] = []
    t = [0.0]
    docs = {f"src:{i}": f"p{i} value" for i in range(5)}

    def reval(artifact_id: str) -> tuple[str, str]:
        calls.append(artifact_id)
        return docs[artifact_id], "v1"

    ret = InMemoryRetriever()
    for aid, text in docs.items():
        ret.add(aid, text)
    cache = _pool(ret, serve_gate=0.5, clock=lambda: t[0],
                  freshness=FreshnessPolicy(max_age=10, revalidate=reval))
    for i in range(5):
        cache.get(f"p{i}")                # five builds, five owners
    synth = cache._synth
    epoch = cache._rows_epoch
    t[0] = 100.0                          # ALL owners expired
    r = cache.get("value")
    assert len(calls) == 3                # the revalidator cap, serving-rank priority
    assert len({p.unit_id for p in r.pool}) == 3   # past-cap owners masked THIS read
    assert all(u.is_fresh for u in cache._units.values())   # ...but never dirtied
    assert cache._rows_epoch == epoch     # revalidate-unchanged: NO epoch bump
    assert synth.calls == 5               # type: ignore[attr-defined]
    r2 = cache.get("value one")           # later read revalidates the remainder
    assert len(calls) == 5
    assert len({p.unit_id for p in r2.pool}) == 5


def test_ttl_revalidate_unchanged_no_epoch_bump() -> None:
    t = [0.0]
    ret = InMemoryRetriever()
    ret.add("src:a", "alpha value")
    cache = _pool(ret, serve_gate=0.3, clock=lambda: t[0],
                  freshness=FreshnessPolicy(max_age=10,
                                            revalidate=lambda a: ("alpha value", "v1")))
    cache.get("alpha value")
    synth = cache._synth
    epoch = cache._rows_epoch
    t[0] = 50.0
    r = cache.get("alpha one")            # expired -> revalidated -> unchanged -> fresh
    assert r.cache_hit is True
    assert cache._rows_epoch == epoch     # mark-fresh only: no rows change, no rebuild
    assert synth.calls == 1               # type: ignore[attr-defined]
    unit = next(iter(cache._units.values()))
    assert unit.is_fresh and unit.freshness_epoch == 50.0


# ---------------------------------------------------------------- serving stack (P8, §3.1)

def test_pool_stage1_width_constant() -> None:
    assert _POOL_STAGE1_RAW == 600
    assert _POOL_STAGE1_WIDTH == 400


def test_pool_candidates_width() -> None:
    cache = _pool()
    n = 450
    basis = [tuple(1.0 if j == i else 0.0 for j in range(n)) for i in range(n)]
    unit = Cognition(
        id="cog:big", namespace="", query="q", query_embedding=(1.0,) + (0.0,) * (n - 1),
        understanding={"claims": [f"claim {i}" for i in range(n)]}, evidence=(),
        provenance=ProvenanceManifest("s", "p"),
        claim_embeddings=tuple(basis),
    )
    cache._units[unit.id] = unit
    rows = [(1.0 - i * 0.001, ClaimRef("cog:big", i, f"claim {i}")) for i in range(n)]
    kept = cache._pool_candidates(rows)
    assert len(kept) == _POOL_STAGE1_WIDTH   # truncated to the rerank window, no dups lost


@pytest.mark.skipif(_np is None, reason="numpy not installed — vector dedup inactive")
def test_dedup_drops_near_identical() -> None:
    # v0.6 (F2): the 0.95 vector collapse is WITHIN-OWNER only — a unit's own
    # rephrasings are redundancy; a cross-owner near-duplicate is CORROBORATION and stays.
    cache = _pool()
    u1 = _mk_unit("cog:u1", "", "alpha", ["alpha value"], "art:u1")
    u1.understanding["claims"] = ["u1 says alpha value", "alpha value per u1"]  # same VECTOR
    u1.claim_embeddings = (tuple(_embed("alpha value")), tuple(_embed("alpha value")))
    cache._units["cog:u1"] = u1
    u2 = _mk_unit("cog:u2", "", "alpha", ["alpha value"], "art:u2")
    u2.understanding["claims"] = ["u2 says alpha value"]
    u2.claim_embeddings = (tuple(_embed("alpha value")),)
    cache._units["cog:u2"] = u2
    rows = [(1.0, ClaimRef("cog:u1", 0, "u1 says alpha value")),
            (0.99, ClaimRef("cog:u1", 1, "alpha value per u1")),
            (0.98, ClaimRef("cog:u2", 0, "u2 says alpha value"))]
    kept = cache._pool_candidates(rows)
    # u1's own rephrasing collapsed; u2's identical-vector copy SURVIVES under its owner.
    assert [(h.unit_id, h.claim_idx) for h in kept] == [("cog:u1", 0), ("cog:u2", 0)]
    # Gold-transfer record: the dropped row maps to the kept row it collapsed into.
    assert cache._last_collapsed == {("cog:u1", 1): ("cog:u1", 0)}


def test_purepython_dedup_text_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache_module, "_np", None)
    cache = _pool()
    for uid in ("cog:u1", "cog:u2"):
        u = _mk_unit(uid, "", "alpha", ["alpha value"], f"art:{uid}")
        u.understanding["claims"] = [f"{uid} says alpha value", "same text"]
        cache._units[uid] = u
    rows = [(1.0, ClaimRef("cog:u1", 0, "cog:u1 says alpha value")),
            (0.99, ClaimRef("cog:u2", 0, "cog:u2 says alpha value"))]
    # Near-identical VECTORS but different text: pure python keeps BOTH (text-only dedup).
    assert len(cache._pool_candidates(rows)) == 2
    # Exact-duplicate TEXT within ONE owner still collapses...
    rows_same_owner = [(1.0, ClaimRef("cog:u1", 0, "same text")),
                       (0.99, ClaimRef("cog:u1", 1, "same text"))]
    kept = cache._pool_candidates(rows_same_owner)
    assert [(h.unit_id, h.claim_idx) for h in kept] == [("cog:u1", 0)]
    assert cache._last_collapsed == {("cog:u1", 1): ("cog:u1", 0)}
    # ...but the SAME text held by two owners is corroboration: both copies survive (F2).
    rows_cross = [(1.0, ClaimRef("cog:u1", 0, "same text")),
                  (0.99, ClaimRef("cog:u2", 0, "same text"))]
    assert [h.unit_id for h in cache._pool_candidates(rows_cross)] == ["cog:u1", "cog:u2"]


def test_rerank_orders_by_score_then_cosine() -> None:
    cache = _pool(reranker=lambda q, texts: [1.0, 2.0, 2.0])
    cand = [_PoolHit(0.9, "u", 0, "a"), _PoolHit(0.8, "u", 1, "b"), _PoolHit(0.7, "u", 2, "c")]
    ordered, reranked, ms = cache._pool_rank("q", cand)
    assert reranked is True and ms is not None
    assert [h.text for h in ordered] == ["b", "c", "a"]   # tie broken by cosine position


def test_rerank_exception_degrades_to_cosine() -> None:
    events: list[dict] = []

    def boom(q: str, texts: list[str]) -> list[float]:
        raise RuntimeError("model exploded")

    cache = _pool(reranker=boom, on_event=events.append)
    cand = [_PoolHit(0.9, "u", 0, "a"), _PoolHit(0.8, "u", 1, "b")]
    ordered, reranked, _ = cache._pool_rank("q", cand)
    assert reranked is False
    assert [h.text for h in ordered] == ["a", "b"]        # cosine order preserved
    assert [e for e in events if e["event"] == "rerank_failed"]


def test_rerank_length_mismatch_degrades() -> None:
    events: list[dict] = []
    cache = _pool(reranker=lambda q, texts: [1.0], on_event=events.append)
    cand = [_PoolHit(0.9, "u", 0, "a"), _PoolHit(0.8, "u", 1, "b")]
    ordered, reranked, _ = cache._pool_rank("q", cand)
    assert reranked is False and [h.text for h in ordered] == ["a", "b"]
    assert "2 texts" in [e for e in events if e["event"] == "rerank_failed"][0]["reason"]


def test_rerank_nan_treated_as_neg_inf() -> None:
    cache = _pool(reranker=lambda q, texts: [float("nan"), 1.0])
    cand = [_PoolHit(0.9, "u", 0, "a"), _PoolHit(0.8, "u", 1, "b")]
    ordered, reranked, _ = cache._pool_rank("q", cand)
    assert reranked is True
    assert [h.text for h in ordered] == ["b", "a"]        # NaN sorts LAST, not first


def test_gates_ignore_reranker() -> None:
    # Byte-identical DECISIONS with and without a (perverse, order-inverting) reranker:
    # rerank output touches serving order only, never serve/build/floor or coverage.
    def build(reranker):  # type: ignore[no-untyped-def]
        events: list[dict] = []
        ret = InMemoryRetriever()
        ret.add("src:a", "alpha one")
        ret.add("src:b", "beta one")
        cache = _pool(ret, serve_gate=0.5, reranker=reranker, on_event=events.append)
        return cache, events

    plain, ev_plain = build(None)
    invert, ev_invert = build(lambda q, texts: list(range(len(texts))))
    results = []
    for cache in (plain, invert):
        results.append([cache.get(q) for q in
                        ("alpha one", "one", "gamma value", "one two")])
    for ra, rb in zip(*results):
        assert (ra.cache_hit, ra.escalated, ra.needs_retrieval) == \
               (rb.cache_hit, rb.escalated, rb.needs_retrieval)
        assert ra.confidence == pytest.approx(rb.confidence)
        assert ra.coverage == pytest.approx(rb.coverage)
    gates_a = [(e["coverage"], e["gate"], e["outcome"])
               for e in ev_plain if e["event"] == "pool_gate"]
    gates_b = [(e["coverage"], e["gate"], e["outcome"])
               for e in ev_invert if e["event"] == "pool_gate"]
    assert gates_a == gates_b
    # ...and the reranker DID engage: some served order differs.
    orders_a = [[p.claim for p in r.pool] for r in results[0]]
    orders_b = [[p.claim for p in r.pool] for r in results[1]]
    assert orders_a != orders_b


def test_pack_never_exceeds_budget() -> None:
    cache = _pool(serve_budget=25)
    hits = [_PoolHit(1.0 - i * 0.01, "u1", i, "x" * 38) for i in range(5)]   # est 10 each
    picked, est, _headers = cache._pool_pack(hits)
    assert len(picked) == 2               # 11 + 10 = 21; a third (31) would overflow
    assert est == 21 and est <= 25


def test_pack_stops_no_skip() -> None:
    cache = _pool(serve_budget=35)
    big = _PoolHit(0.9, "u1", 0, "y" * 118)    # est("- "+118)=30, +1 group = 31
    small = _PoolHit(0.8, "u1", 1, "z" * 18)   # est 5 -> 36 > 35
    picked, _est, _ = cache._pool_pack([big, small])
    assert [h.claim_idx for h in picked] == [0]   # STOP at first overflow — no skip-scan


def test_at_least_one_claim() -> None:
    events: list[dict] = []
    cache = _pool(serve_budget=10, on_event=events.append)
    huge = _PoolHit(0.9, "u1", 0, "w" * 200)
    picked, est, _ = cache._pool_pack([huge])
    assert len(picked) == 1               # served anyway
    overruns = [e for e in events if e["event"] == "pool_budget_overrun"]
    assert overruns and overruns[0]["est_tokens"] == est


def test_pack_counts_headers_once_per_unit() -> None:
    unit = _mk_unit("u1", "", "alpha", ["alpha"], "art:a")
    cache = _pool(serve_budget=14, pool_header=lambda u: "HDR!HDR!")   # header est 2
    cache._units["u1"] = unit
    hits = [_PoolHit(0.9, "u1", 0, "c" * 18), _PoolHit(0.8, "u1", 1, "d" * 18)]  # est 5 each
    picked, est, headers = cache._pool_pack(hits)
    assert len(picked) == 2               # 5+1+(2+1) + 5 = 14: header counted ONCE
    assert est == 14 and headers["u1"] == "HDR!HDR!"
    tight = _pool(serve_budget=13, pool_header=lambda u: "HDR!HDR!")
    tight._units["u1"] = unit
    assert len(tight._pool_pack(hits)[0]) == 1
    # A raising header hook is swallowed: packs (and renders) headerless, never crashes.
    def bad_header(u: Cognition) -> str:
        raise RuntimeError("boom")
    grumpy = _pool(serve_budget=13, pool_header=bad_header)
    grumpy._units["u1"] = unit
    picked3, _est3, headers3 = grumpy._pool_pack(hits)
    assert len(picked3) == 2 and headers3["u1"] == ""
    assert "HDR" not in SemanticCache._pool_render(picked3, headers3)


def test_group_order_by_best_rank_and_within_group_score_order() -> None:
    picked = [_PoolHit(0.9, "A", 0, "a1"), _PoolHit(0.8, "B", 0, "b2"),
              _PoolHit(0.7, "A", 1, "a3")]
    rendered = SemanticCache._pool_render(picked, {})
    # Group A first (best rank), its claims in SERVING order; then group B; blank-line joined.
    assert rendered == "- a1\n- a3\n\n- b2"
    with_headers = SemanticCache._pool_render(picked, {"A": "## A", "B": "## B"})
    assert with_headers == "## A\n- a1\n- a3\n\n## B\n- b2"


def test_pool_dedupe_claims() -> None:
    # v0.6 (F2, supersedes the D2 single-copy collapse): the same claim text cached by
    # two SOURCES serves BOTH copies — per-source questions need the owner's own attributed
    # copy; only a unit's OWN duplicates collapse (pinned above).
    ret = InMemoryRetriever()
    ret.add("src:a", "target value")
    ret.add("src:b", "target value")
    cache = _pool(ret, serve_gate=0.4)
    cache.get("target value")
    assert len(cache._units) == 2
    r = cache.get("target one")
    assert r.understanding["claims"].count("target value") == 2
    assert len({p.unit_id for p in r.pool if p.claim == "target value"}) == 2


def test_summary_rows_excluded_from_coverage_and_pack() -> None:
    events: list[dict] = []
    ret = InMemoryRetriever()
    ret.add("src:a", "target unique value")
    cache = _pool(ret, serve_gate=0.4, on_event=events.append)
    cache.get("target unique value")      # unit: claims=["target unique value"], summary too
    # Poison the claim rows only; the summary stays highly similar to the next query.
    unit = next(iter(cache._units.values()))
    unit.understanding["claims"] = ["junk"]
    unit.claim_embeddings = (tuple(_embed("junk")), unit.claim_embeddings[-1])
    cache._rows_epoch += 1
    cache._index_unit_rows(unit)
    r = cache.get("target unique one")    # summary cos ~0.67 — but summaries are NOT rows
    gate_ev = [e for e in events if e["event"] == "pool_gate"][-1]
    assert gate_ev["outcome"] == "build"  # cov read atomic rows only
    assert gate_ev["coverage"] < 0.1
    # The summary text never serves as a CLAIM ROW (the "## ..." header line legitimately
    # carries the build query since the §7.1 fork — rows are the assertion surface).
    assert "- target unique value" not in r.context["pool"]
    assert all(p.claim != "target unique value" for p in r.pool)


def test_summary_only_unit_zero_pool_coverage() -> None:
    class _SummaryOnlySynth:
        def synthesize(self, query: str, chunks: list[Chunk]) -> Synthesis:
            return Synthesis(understanding={"summary": " ".join(c.text for c in chunks)},
                             used=list(range(len(chunks))))

    events: list[dict] = []
    ret = InMemoryRetriever()
    ret.add("src:a", "target value")
    cache = _pool(ret, _SummaryOnlySynth(), serve_gate=0.4, on_event=events.append)
    cache.get("target value")
    cache.get("target one")               # nothing atomic to serve: coverage must be 0.0
    gate_ev = [e for e in events if e["event"] == "pool_gate"][-1]
    assert gate_ev["coverage"] == 0.0
    assert cache.stats()["pool_claims_total"] == 0


def test_served_claim_owner_attribution_exact() -> None:
    ret = InMemoryRetriever()
    ret.add("src:a", "alpha one")
    ret.add("src:b", "beta one")
    ret.add("src:c", "target value")
    cache = _pool(ret, serve_gate=0.4,
                  reranker=lambda q, texts: list(range(len(texts))))   # inverted order
    cache.get("alpha one")
    cache.get("target value")
    r = cache.get("one two")
    assert r.pool
    for p in r.pool:                      # after dedup + inverting rerank: exact ownership
        owner = cache._units[p.unit_id]
        assert p.claim in owner.understanding["claims"]


# ------------------------------------------------- serving hardening (v0.6 F1/F2/F4)

def test_pool_without_header_warns_once_at_construction() -> None:
    # UX guard (measured ladder n=605: bare default 0.6777 vs metadata callable 0.7306 —
    # the gap is per-source questions honestly refusing): opting into the pool path
    # without a pool_header must WARN at construction, never silently under-attribute.
    import warnings as _w
    with pytest.warns(UserWarning, match="pool_header"):
        _pool(InMemoryRetriever())
    with _w.catch_warnings():                       # header callable -> no warning
        _w.simplefilter("error")
        _pool(InMemoryRetriever(), pool_header=lambda u: "[hdr]")
    with _w.catch_warnings():                       # explicit unit path -> no warning
        _w.simplefilter("error")
        SemanticCache(InMemoryRetriever(), _EchoSynth(),  # type: ignore[arg-type]
                      embedder=FunctionEmbedder(_embed), read_path="unit")


def test_pool_payload_always_attributed() -> None:
    # F1 + the header ladder (measured n=605 strict: opaque "[source: art:N]" ids 0.6413,
    # "## query" 0.6777, the metadata header 0.7306): with pool_header=None the library
    # serves the v0.7 ladder — ingest meta "[{title} | {source} | {date}]" first, else
    # "## " + unit.query[:60] (the build query carries source-identifying text), falling
    # back to "[source: {artifact_id}]" ONLY when the unit has no query text. It is still
    # UNABLE to serve an unattributed pool payload: a header line opens EVERY group.
    ret = InMemoryRetriever()
    ret.add("src:a", "alpha value one")
    ret.add("src:b", "beta value two")
    cache = _pool(ret, serve_gate=0.4)                 # NO pool_header
    cache.get("alpha value one")                       # builds both artifacts
    r = cache.get("value one")
    groups = r.context["pool"].split("\n\n")
    assert len(groups) == 2
    for g in groups:                                   # header line, then the claims
        assert g.splitlines()[0] == "## alpha value one"   # both units' birth query
        assert g.splitlines()[1].startswith("- ")
    # RUNG 1 — ingest metadata outranks the query rung; missing keys are skipped.
    uid = r.pool[0].unit_id
    cache._units[uid].source_meta = {"title": "T", "source": "S", "date": "2026-08-11"}
    assert cache._pool_header_text(uid) == "[T | S | 2026-08-11]"
    cache._units[uid].source_meta = {"title": "T", "date": "2026-08-11"}
    assert cache._pool_header_text(uid) == "[T | 2026-08-11]"
    # All-empty meta values fall THROUGH the rung (never render "[]").
    cache._units[uid].source_meta = {"title": " ", "extra": "x"}
    assert cache._pool_header_text(uid).startswith("## ")
    cache._units[uid].source_meta = {}
    # RUNG 2 — the query prefix truncates at 60 chars.
    long_q = "q" * 100
    cache._units[uid].query = long_q
    assert cache._pool_header_text(uid) == "## " + "q" * 60
    # RUNG 3 — a query-less unit falls back to the artifact id; evidence-less to its own id.
    cache._units[uid].query = ""
    assert cache._pool_header_text(uid) in ("[source: src:a]", "[source: src:b]")
    bare = _mk_unit("cog:bare", "", "", ["gamma"], "art:x")
    bare.evidence = ()
    cache._units["cog:bare"] = bare
    assert cache._pool_header_text("cog:bare") == "[source: cog:bare]"
    # An explicit callback still overrides the ENTIRE ladder, meta included.
    ret2 = InMemoryRetriever()
    ret2.add("src:a", "alpha value one", meta={"title": "T"})
    custom = _pool(ret2, serve_gate=0.4, pool_header=lambda u: "## MINE")
    custom.get("alpha value one")
    assert custom.get("alpha one").context["pool"].startswith("## MINE\n")


def test_dedup_preserves_cross_owner_copies() -> None:
    # F2 end-to-end: two sources assert the SAME fact; both copies serve, each under its
    # own attribution — corroboration (and per-source answers) survive the 0.95 collapse.
    ret = InMemoryRetriever()
    ret.add("src:a", "target value")
    ret.add("src:b", "target value")
    cache = _pool(ret, serve_gate=0.4)
    cache.get("target value")
    r = cache.get("target one")
    copies = [p for p in r.pool if p.claim == "target value"]
    assert len(copies) == 2 and len({p.unit_id for p in copies}) == 2
    groups = r.context["pool"].split("\n\n")
    assert len(groups) == 2
    for g in groups:                                   # §7.1 default: "## <query-prefix>"
        assert g.splitlines()[0] == "## target value"
        assert "- target value" in g                   # the fact under EACH owner's header
    # Distinct per-owner queries render distinct per-owner headers on the next serve.
    for i, p in enumerate({c.unit_id for c in copies}):
        cache._units[p].query = f"outlet {i} coverage"
    r2 = cache.get("target one")
    heads = {g.splitlines()[0] for g in r2.context["pool"].split("\n\n")}
    assert heads == {"## outlet 0 coverage", "## outlet 1 coverage"}


def test_escalation_raw_is_attributed() -> None:
    # F4: the RAG floor's raw chunks carry the same [source: ...] attribution the pool
    # payload does — escalation must not reopen the outlet-blindness hole.
    ret = InMemoryRetriever()
    ret.add("src:a", "alpha one")
    cache = _pool(ret, serve_gate=0.9, coverage_floor=0.9)
    r = cache.get("alpha two")                         # build; post-build cov < floor
    assert r.escalated is True
    assert r.context["raw"] == ["[source: src:a]\nalpha one"]
    # CONTEXT_RAW serves the same attributed form (one payload surface, one format).
    ret2 = InMemoryRetriever()
    ret2.add("src:a", "alpha one")
    raw_cache = _pool(ret2, serve_gate=0.9, strategy="context_raw")
    r2 = raw_cache.get("alpha two")
    assert r2.context["raw"] == ["[source: src:a]\nalpha one"]


# ------------------------------------------------------------------- escalation / S2 (P6/P7)

def test_escalation_floor_appends_raw() -> None:
    ret = _CountingRetriever()
    ret.add("src:a", "alpha one")
    cache = _pool(ret, serve_gate=0.9, coverage_floor=0.9)
    r = cache.get("alpha two")            # build; post-build coverage 0.5 < floor
    assert r.cache_hit is False and r.escalated is True
    # The P5 probe chunks — reused, not re-fetched — served WITH attribution (F4).
    assert r.context["raw"] == ["[source: src:a]\nalpha one"]
    assert ret.calls == 1                 # the single retrieval fed build AND floor
    assert r.needs_retrieval is True


def test_s2_demoted_serve_still_gets_rag_floor() -> None:
    ret = _CountingRetriever()
    ret.add("src:a", "alpha one")
    cache = _pool(ret, serve_gate=0.3, coverage_floor=0.5, coverage_ceiling=1.0,
                  coverage_scorer=lambda q, u: 0.0)
    cache.get("alpha one")                # build (post-build cov 1.0 escapes the S2 band)
    n = ret.calls
    r = cache.get("alpha two")            # pure serve (cov 0.5 >= gate) -> S2 demotes to 0.0
    assert r.cache_hit is True            # still a zero-synthesis read
    assert r.escalated is True            # ...but the RAG floor MUST fire (v0.5 contract)
    assert r.context["raw"] == ["[source: src:a]\nalpha one"]   # attributed raw (F4)
    assert ret.calls == n + 1             # the retrieve-NOW was this read's single retrieval


def test_s2_scorer_on_served_payload() -> None:
    seen: dict[str, object] = {}

    def scorer(query: str, understanding: dict) -> float:
        seen["u"] = understanding
        return 0.9

    ret = InMemoryRetriever()
    ret.add("src:a", "alpha value")
    cache = _pool(ret, serve_gate=0.3, coverage_floor=0.5, coverage_ceiling=1.0,
                  coverage_scorer=scorer)
    cache.get("alpha value")
    r = cache.get("alpha one")            # cov 0.5 sits in [floor, ceiling) -> scorer runs
    assert seen["u"] == {"claims": r.understanding["claims"]}   # judges what actually serves
    assert r.coverage == 0.9 and r.escalated is False


def test_floor_raw_capped_but_never_empty() -> None:
    ret = _CountingRetriever()
    ret.add("src:a", "alpha " + "value " * 40)   # one chunk, est far over the budget
    cache = _pool(ret, serve_gate=0.99, coverage_floor=0.99, serve_budget=5)
    r = cache.get("alpha two")
    assert r.escalated is True
    assert len(r.context["raw"]) == 1     # capped to the budget, but never empty


# ------------------------------------------------------------------- result & event contract

def test_result_fields_pool_mode() -> None:
    ret = InMemoryRetriever()
    ret.add("src:a", "alpha value one")
    cache = _pool(ret, serve_gate=0.5)
    cache.get("alpha value one")
    r = cache.get("alpha one")
    assert r.recalled == []               # unit-path field, empty on the pool path
    assert r.evidence == []
    assert set(r.context) == {"pool", "serve", "understanding"}
    assert r.pool and r.pool[0].unit_id == r.unit_id      # top-served owner
    assert r.pool[0].score == pytest.approx(r.confidence)  # score = query cosine, always
    assert r.confidence == pytest.approx(2.0 / 6.0 ** 0.5)  # cov0 = cos({alpha,one},{alpha,value,one})
    assert r.needs_retrieval is False


def test_pool_understanding_has_no_summary() -> None:
    ret = InMemoryRetriever()
    ret.add("src:a", "alpha value one")
    cache = _pool(ret, serve_gate=0.5)
    cache.get("alpha value one")
    r = cache.get("alpha one")
    assert "summary" not in r.understanding
    assert set(r.understanding) == {"claims"}
    assert "summary" not in r.context["understanding"]


def test_pool_hit_tokens_saved_credits_top_owner_only() -> None:
    class _TieredUsageSynth:
        def __init__(self) -> None:
            self.n = 0

        def synthesize(self, query: str, chunks: list[Chunk]) -> Synthesis:
            self.n += 1
            text = " ".join(c.text for c in chunks)
            return Synthesis(understanding={"summary": text, "claims": [text]},
                             used=list(range(len(chunks))),
                             usage=Usage(prompt_tokens=100 if self.n == 1 else 40))

    ret = InMemoryRetriever()
    ret.add("src:a", "alpha one")
    ret.add("src:b", "beta two")
    cache = _pool(ret, _TieredUsageSynth(), serve_gate=0.5)
    cache.get("alpha one")                # build A: 100 tokens
    cache.get("beta two")                 # build B: 40 tokens
    assert cache.stats()["tokens_saved"] == 0
    r = cache.get("alpha")                # serve; top owner is A
    assert r.cache_hit is True and r.pool[0].claim == "alpha one"
    assert cache.stats()["tokens_saved"] == 100   # top owner ONLY — conservative crediting


def test_pool_build_read_saves_zero_tokens() -> None:
    ret = InMemoryRetriever()
    ret.add("src:a", "alpha one")
    cache = SemanticCache(ret, _UsageSynth(prompt=50, completion=10),
                          embedder=FunctionEmbedder(_embed),
                          read_path="pool", serve_gate=0.5, coverage_floor=0.0)
    cache.get("alpha one")
    assert cache.stats()["tokens_saved"] == 0


def test_pool_served_event_schema() -> None:
    events: list[dict] = []
    ret = InMemoryRetriever()
    ret.add("src:a", "alpha value one")
    cache = _pool(ret, serve_gate=0.5, on_event=events.append)
    cache.get("alpha value one")
    cache.get("alpha one")
    served = [e for e in events if e["event"] == "pool_served"]
    assert len(served) == 2               # every pool read serves (and reports) a payload
    expected = {"event", "ts", "n_claims", "n_units", "coverage", "gate", "est_tokens",
                "budget", "candidates_raw", "candidates_unique", "reranked", "rerank_ms",
                "reuse"}
    assert set(served[-1]) == expected
    assert served[-1]["reranked"] is False and served[-1]["rerank_ms"] is None
    assert served[-1]["budget"] == 1000


def test_stats_pool_extensions() -> None:
    pool_keys = {"read_path", "serve_budget", "serve_gate", "serve_gate_effective",
                 "pool_noise_ceiling", "pool_claims_fresh", "pool_claims_total",
                 "pool_mask_rate", "pool_serves", "probe_reads", "admission_reuses",
                 "retrievals", "rebuilds_by_read", "builds_by_gap", "avg_units_per_serve",
                 "pool_scan_slow"}
    assert pool_keys <= set(_pool().stats())
    unit_cache = SemanticCache(InMemoryRetriever(), _EchoSynth(),   # type: ignore[arg-type]
                               embedder=FunctionEmbedder(_embed), read_path="unit")
    assert not (pool_keys & set(unit_cache.stats()))                # unit stats unchanged


# ----------------------------------------------------------------------- v0.5 bit-identity

def test_unit_mode_bit_identical() -> None:
    """v0.7 BREAKING flip: the bare default no longer pins "unit" — the ESCAPE HATCH does.

    Explicit ``read_path="unit"`` is the byte-identical pre-flip unit path, and under the
    lexical ``HashingEmbedder`` the ``read_path=None`` sentinel resolves onto exactly that
    same unit path (the loud fallback warning itself is pinned in
    test_v07_default_read_path.py)."""
    def mk(embedder: object, **kw: object) -> SemanticCache:
        ret = InMemoryRetriever()
        ret.add("src:a", "alpha value one")
        ret.add("src:b", "beta value two")
        return SemanticCache(ret, _Synth(), embedder=embedder,       # type: ignore[arg-type]
                             hit_threshold=0.6, coverage_floor=0.3,
                             clock=lambda: 1000.0, **kw)   # type: ignore[arg-type]

    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")            # the loud fallback warning, pinned elsewhere
        resolved = mk(HashingEmbedder())     # sentinel default -> resolves to "unit"
    explicit = mk(HashingEmbedder(), read_path="unit")
    for q in ("alpha value one", "alpha one", "beta value two", "gamma", "alpha value one"):
        assert resolved.get(q) == explicit.get(q)
    resolved.source_changed("src:a", text="alpha revised")
    explicit.source_changed("src:a", text="alpha revised")
    assert resolved.get("alpha value one") == explicit.get("alpha value one")
    assert resolved.stats() == explicit.stats()

    # Explicit unit under a SEMANTIC embedder: still the v0.5 unit machinery — pool
    # machinery is UNREACHABLE (nothing hydrated, nothing counted), stats carry no pool keys.
    unit = mk(FunctionEmbedder(_embed), read_path="unit")
    for q in ("alpha value one", "alpha one", "beta value two", "gamma", "alpha value one"):
        unit.get(q)
    assert unit._claim_indexes == {}
    assert unit._pool_hydrated == set()
    assert unit._pool_serves == 0 and unit._probe_reads == 0
    assert "pool_serves" not in unit.stats()
