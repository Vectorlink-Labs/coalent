"""MECHANISM 2 (v0.7) — first-pass query decomposition (opt-in ``decompose=``).

PREREG-MECH2.md, locked 2026-08-17. The evidence: the FRAMES X-ray's dominant gap is
ranking — 82/146 misses have the gold IN the claim pool but unreached because the
compositional query vector points away from it (gold full-pool rank p50 104 / p95 8,344).
The mechanism: when ``decompose=`` is armed with a BYO callable (the library NEVER calls
an LLM itself), each pool read embeds the callable's 2-4 sub-questions (optionally with a
HyDE-lite hypothetical concat) and scores every claim as the MAX cosine over
{raw query, all probes}. Everything downstream — top-600 scan width, within-owner dedup,
budget pack, gate, headers — is unchanged.

The safety properties under test everywhere here: ``decompose=False`` (the default) is
byte-identical v0.6 behavior; the raw query is ALWAYS in the union (arm-B tail collapse
is pre-refuted); a failing/malformed callable degrades to the raw query (fail open, never
crash the read); the budget contract holds under the union; first pass only (the
behavioral loop never re-invokes the callable); serde is untouched (no new persisted
state); the ship-plan embed-batching (raw query + probes in ONE embedder call,
PREREG-MECH2B) is answer-neutral — byte-identical payload sha1 and scores vs the
per-text path.
"""
from __future__ import annotations

import hashlib
import re

import pytest

from coalent import FunctionEmbedder
from coalent.semantic import Chunk, SemanticCache, Synthesis
from coalent.semantic.cache import _DECOMPOSE_MAX_SUBS, _est_tokens
from coalent.semantic.serde import cognition_from_dict, cognition_to_dict

_AXES = ("alpha", "beta", "gamma", "kappa", "one", "two", "three", "value", "junk")
_STRIP = re.compile(r"[^\w\s]")


def _embed(text: str) -> list[float]:
    words = set(_STRIP.sub(" ", text.lower()).split())
    v = [1.0 if a in words else 0.0 for a in _AXES]
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v] if n else v


class _RecordingEmbedder(FunctionEmbedder):
    """FunctionEmbedder that records every call — ``embed_many`` batches (the read's
    batched query+probes path) and any stray single ``embed`` calls."""

    def __init__(self) -> None:
        super().__init__(_embed)
        self.many_batches: list[list[str]] = []
        self.single_calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.single_calls.append(text)
        return list(_embed(text))

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        self.many_batches.append(list(texts))
        return [list(_embed(t)) for t in texts]


class _PerTextOnlyEmbedder:
    """The pre-batching reference arm: NO ``embed_many`` attribute at all, so the read
    embeds the raw query and each probe text in its OWN ``embed`` call — exactly the
    old (pre-ship-optimization) read path's embedder-call shape."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return list(_embed(text))


class _ClaimSynth:
    """Fixed (mutable between calls) claim list — deterministic build content."""

    def __init__(self, claims: list[str]) -> None:
        self.claims = list(claims)
        self.calls = 0

    def synthesize(self, query: str, chunks: list[Chunk]) -> Synthesis:
        self.calls += 1
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
    kw.setdefault("clock", lambda: 1_000.0)
    return SemanticCache(retriever, synth,                            # type: ignore[arg-type]
                         read_path="pool", **kw)                      # type: ignore[arg-type]


def _two_unit_world(events: list[dict] | None = None,
                    **kw: object) -> tuple[SemanticCache, str, str]:
    """Unit A (src:a, claim 'alpha value') + unit B (src:b, claim 'beta kappa two').
    The default tiny ``serve_budget`` (10) packs exactly ONE owner group per read, so
    whichever claim RANKS FIRST is the whole payload — burial is a ranking fact."""
    ret = _WordRetriever()
    ret.set("src:a", "alpha value gamma.")
    ret.set("src:b", "beta kappa two.")
    synth = _ClaimSynth(["alpha value"])
    kw.setdefault("serve_budget", 10)
    cache = _pool(ret, synth, on_event=(events.append if events is not None else None),
                  **kw)
    ra = cache.get("alpha value")                       # build unit A
    synth.claims = ["beta kappa two"]
    rb = cache.get("beta kappa")                        # build unit B
    assert ra.unit_id != rb.unit_id
    return cache, ra.unit_id, rb.unit_id


def _serving_fingerprint(r) -> tuple:  # type: ignore[no-untyped-def]
    return (r.context["pool"], [c.claim for c in r.pool],
            [c.unit_id for c in r.pool], r.coverage, r.confidence,
            r.cache_hit, r.escalated, r.needs_retrieval)


# ------------------------------------------------------------------ constructor contract

def test_decompose_constructor_contract() -> None:
    ret = _WordRetriever()
    synth = _ClaimSynth(["alpha"])
    # non-callable truthy values are a contract error, LOUD
    with pytest.raises(TypeError, match="decompose must be False or a callable"):
        _pool(ret, synth, decompose=True)
    with pytest.raises(TypeError, match="decompose must be False or a callable"):
        _pool(ret, synth, decompose=42)
    # armed on the unit path the knob would be structurally inert — fail LOUD
    with pytest.raises(ValueError, match="decompose requires read_path='pool'"):
        SemanticCache(ret, synth, embedder=FunctionEmbedder(_embed),  # type: ignore[arg-type]
                      decompose=lambda q: [], read_path="unit")
    # the valid arming shapes construct fine
    _pool(ret, synth, decompose=lambda q: [])
    _pool(ret, synth, decompose=False)


# ------------------------------------------------------------- default-OFF byte-inert

def test_default_off_byte_inert() -> None:
    # THE prereg gauntlet pin: same store + query -> byte-identical payload with
    # decompose=False vs the knob absent, over every serving observable.
    cache_absent, _, _ = _two_unit_world()
    cache_false, _, _ = _two_unit_world(decompose=False)
    for q in ("alpha value one", "beta kappa", "alpha gamma", "kappa two three"):
        assert _serving_fingerprint(cache_absent.get(q)) \
            == _serving_fingerprint(cache_false.get(q))


def test_fail_open_degrades_to_raw() -> None:
    # A raising callable, a malformed return, and an empty return all serve the raw-query
    # payload byte-identically — the read NEVER crashes and never changes.
    def _boom(q: str) -> list[dict]:
        raise RuntimeError("BYO decomposer down")

    cache_off, _, _ = _two_unit_world()
    worlds = [
        _two_unit_world(decompose=_boom)[0],
        _two_unit_world(decompose=lambda q: "not a list")[0],
        _two_unit_world(decompose=lambda q: [{"no_q": "x"}, {"q": "   "}])[0],
        _two_unit_world(decompose=lambda q: [])[0],
    ]
    for q in ("alpha value one", "beta kappa"):
        base = _serving_fingerprint(cache_off.get(q))
        for cache in worlds:
            assert _serving_fingerprint(cache.get(q)) == base
    # the sanctioned empty return emits no pool_decomposed event
    ev: list[dict] = []
    cache_empty, _, _ = _two_unit_world(ev, decompose=lambda q: [])
    cache_empty.get("alpha value one")
    assert not [e for e in ev if e["event"] == "pool_decomposed"]


# ------------------------------------------------------------------ raw always in union

def test_raw_always_in_union() -> None:
    # Probes that match NOTHING leave the serving byte-identical to OFF (union max with
    # an all-miss probe is the raw scores)...
    cache_off, _, _ = _two_unit_world()
    cache_junk, _, _ = _two_unit_world(decompose=lambda q: [{"q": "junk", "hyde": None}])
    for q in ("alpha value one", "beta kappa"):
        assert _serving_fingerprint(cache_junk.get(q)) \
            == _serving_fingerprint(cache_off.get(q))
    # ...and with budget for both owners, a probe that pulls B in NEVER pushes the raw
    # query's own top claim out: the raw query is always in the union.
    cache_both, _, _ = _two_unit_world(
        serve_budget=60,
        decompose=lambda q: [{"q": "beta kappa two", "hyde": None}])
    r = cache_both.get("alpha value one")
    assert "alpha value" in r.context["pool"]           # raw top claim still served
    assert "beta kappa two" in r.context["pool"]        # probe-found claim added


# ------------------------------------------------------- the buried claim surfaces

def test_buried_claim_surfaces_under_matching_subquery() -> None:
    # Deterministic fake-embedder pin of the mechanism itself: under the raw query the
    # gold-bearing claim ranks below the pack cutoff; a matching sub-question probe
    # lifts it to the top of the SAME pack.
    cache_off, _, _ = _two_unit_world()
    r_off = cache_off.get("alpha value one")
    assert "beta kappa two" not in r_off.context["pool"]          # buried when off
    assert [c.claim for c in r_off.pool] == ["alpha value"]

    ev: list[dict] = []
    cache_on, _, b_on = _two_unit_world(
        ev, decompose=lambda q: [{"q": "beta kappa two", "hyde": None}])
    r_on = cache_on.get("alpha value one")
    assert "beta kappa two" in r_on.context["pool"]               # surfaced when armed
    assert r_on.pool[0].claim == "beta kappa two"
    assert r_on.pool[0].unit_id == b_on
    dec = [e for e in ev if e["event"] == "pool_decomposed"]
    assert dec and dec[-1]["n_subs"] == 1 and dec[-1]["n_probes"] == 1 \
        and dec[-1]["n_hyde"] == 0


def test_hyde_concat_probe_lifts_claim() -> None:
    # HyDE-lite: the probe embeds the "<q> <hyde>" CONCAT — here the sub-question alone
    # misses (junk axis) and only the hypothetical's words reach the buried claim.
    cache, _, b = _two_unit_world(
        decompose=lambda q: [{"q": "junk", "hyde": "beta kappa two"}])
    r = cache.get("alpha value one")
    assert r.pool[0].claim == "beta kappa two"
    assert r.pool[0].unit_id == b


# ------------------------------------------------------------------ budget respected

def test_budget_respected_under_union() -> None:
    ev: list[dict] = []
    # query-gated so the world-construction builds run undecomposed
    cache, _, _ = _two_unit_world(
        ev, decompose=lambda q: [{"q": "beta kappa two", "hyde": None},
                                 {"q": "alpha value", "hyde": None}]
        if q == "alpha gamma one" else [])
    r = cache.get("alpha gamma one")
    served = [e for e in ev if e["event"] == "pool_served"]
    assert served
    last = served[-1]
    assert last["est_tokens"] <= last["budget"] == 10
    # the payload's own estimate agrees with the packer's accounting contract
    assert r.context["pool"] == "" or _est_tokens(r.context["pool"]) <= 3 * last["budget"]


# ------------------------------------------- callable contract: clamp, concat, one call

def test_probe_clamp_concat_and_single_call_per_read() -> None:
    calls: list[str] = []

    def decomp(q: str) -> list[dict]:
        calls.append(q)
        if q != "alpha value one":                     # world builds run undecomposed
            return []
        return [{"q": "beta kappa", "hyde": "beta kappa two"},   # concat embeds
                {"q": "alpha gamma"},                            # hyde absent -> q alone
                {"q": "", "hyde": "dropped"},                    # empty q skipped
                {"q": "kappa one", "hyde": None},
                {"q": "value three", "hyde": ""},                # blank hyde -> q alone
                {"q": "beyond the clamp"}]                       # 5th valid entry: dropped

    emb = _RecordingEmbedder()
    cache, _, _ = _two_unit_world(embedder=emb, residual_spans=True, decompose=decomp)
    emb.many_batches.clear()
    calls.clear()                                       # world construction reads too
    r = cache.get("alpha value one")
    assert calls == ["alpha value one"]                 # exactly ONE call per read
    probe_batches = [b for b in emb.many_batches
                     if "beta kappa beta kappa two" in b]
    assert probe_batches == [
        ["alpha value one",                 # the raw query leads the ONE batched call
         "beta kappa beta kappa two", "alpha gamma", "kappa one"]]
    # _DECOMPOSE_MAX_SUBS caps the ENTRIES considered (4), so the 4th valid entry
    # ("value three") sits beyond the clamp once the empty-q entry consumed a slot.
    # (+1 = the raw query riding the same batch — the ship-plan embed-batching.)
    assert all(len(b) <= _DECOMPOSE_MAX_SUBS + 1 for b in probe_batches)
    # FIRST PASS ONLY: the behavioral loop never re-invokes the callable.
    cache.report_refusal(r.read_id)
    assert calls == ["alpha value one"]


# ------------------------------------------- ship-plan embed-batching: answer-neutral

def test_batched_embed_scores_byte_identical_to_per_text() -> None:
    # THE ship-plan pin (PREREG-MECH2B: "embed-batching optimization applied at merge,
    # answer-neutral, verified byte-identical scores"): same store + query + fake
    # deterministic embedder must serve a byte-identical payload (sha1) with identical
    # scores whether the read's texts embed in ONE batched call (the shipped path) or
    # in per-text calls (the old path's embedder-call shape, forced by an embedder with
    # no ``embed_many``). Latency is the only sanctioned difference.
    def decomp(q: str) -> list[dict]:
        if q != "alpha value one":                     # world builds run undecomposed
            return []
        return [{"q": "beta kappa two", "hyde": None},  # lifts unit B into the pack
                {"q": "gamma", "hyde": "alpha value"}]  # HyDE concat rides the batch

    emb_batch = _RecordingEmbedder()
    emb_pertext = _PerTextOnlyEmbedder()
    cache_b, _, _ = _two_unit_world(embedder=emb_batch, serve_budget=60,
                                    decompose=decomp)
    cache_p, _, _ = _two_unit_world(embedder=emb_pertext, serve_budget=60,
                                    decompose=decomp)
    emb_batch.many_batches.clear()
    emb_batch.single_calls.clear()
    emb_pertext.calls.clear()

    q = "alpha value one"
    r_b = cache_b.get(q)
    r_p = cache_p.get(q)

    # 1) the equivalence: identical payload bytes, identical scores, identical serving
    sha_b = hashlib.sha1(r_b.context["pool"].encode("utf-8")).hexdigest()
    sha_p = hashlib.sha1(r_p.context["pool"].encode("utf-8")).hexdigest()
    assert sha_b == sha_p
    assert [(c.claim, c.unit_id, c.score) for c in r_b.pool] \
        == [(c.claim, c.unit_id, c.score) for c in r_p.pool]
    assert _serving_fingerprint(r_b) == _serving_fingerprint(r_p)
    assert "beta kappa two" in r_b.context["pool"]      # probes actually shaped the pack

    # 2) the call-shape proof this compared something real: the batched arm embedded
    # query + probes in ONE embedder call and never per-text; the reference arm embedded
    # the query and each probe text in separate calls (the pre-batching shape).
    probe_texts = ["beta kappa two", "gamma alpha value"]
    assert emb_batch.many_batches == [[q, *probe_texts]]
    assert emb_batch.single_calls == []
    assert emb_pertext.calls == [q, *probe_texts]


# ------------------------------------------------------------------ serde-neutral

def test_serde_round_trip_untouched() -> None:
    # The knob is runtime config, never persisted state: an armed cache's units
    # serialize byte-identically to an OFF cache's, and the round-trip is unchanged.
    cache_off, a_off, b_off = _two_unit_world()
    cache_on, a_on, b_on = _two_unit_world(
        decompose=lambda q: [{"q": "beta kappa two", "hyde": None}])
    cache_on.get("alpha value one")                     # an armed read mutates nothing
    # read-counter + wall-clock noise only (created_at is a dataclass default, not clock=)
    wash = {"hit_queries": [], "hits": 0, "last_access": 0.0, "created_at": 0.0}
    for uid_off, uid_on in ((a_off, a_on), (b_off, b_on)):
        assert uid_off == uid_on                        # ids are content-deterministic
        d_off = cognition_to_dict(cache_off._units[uid_off]) | wash
        d_on = cognition_to_dict(cache_on._units[uid_on]) | wash
        assert d_off == d_on                            # no new persisted state, ever
        rt = cognition_to_dict(cognition_from_dict(d_on))
        assert rt == cognition_to_dict(cognition_from_dict(d_off))
        assert rt == d_on                               # the round-trip is unchanged
