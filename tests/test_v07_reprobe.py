"""v0.7 Phase B increment 5 — ``reprobe(read_id, hint)``, ITER as an explicit method.

PIPELINE-DESIGN-v07 §THE STACK item 8 / lab arm J: mechanical proper-noun harvest from
the read's SERVED claims (unit-diverse first, question-token filtered — conventions
verbatim), probes ``"<entity> — <sub-question-tail>"`` (+ the caller's ``hint`` as one
extra probe), ONE batched embed, MAX-union re-rank {raw query, original probes, new
probes} over the unchanged pool, repack, fresh Result with a new ``read_id`` linked via
``parent_read_id``. Embeds only, no LLM, no retrieval, no build — and it NEVER
auto-fires: explicit app call, byte-inert when never called.

Pinned here: the planted buried claim reachable ONLY via the entity probe serves on
reprobe · no-entities self-skip · the union includes the originals (nothing served can
score worse) · ONE batched embed · hint as an extra probe · fresh/linked read ids ·
reprobe mutates nothing (a repeat get() is byte-identical) · unit-path fail-loud.
"""
from __future__ import annotations

import re

import pytest

from coalent import FunctionEmbedder
from coalent.semantic import Chunk, SemanticCache, Synthesis

_AXES = ("alpha", "beta", "gamma", "kappa", "one", "two", "three", "value", "junk")
_STRIP = re.compile(r"[^\w\s]")


def _embed(text: str) -> list[float]:
    words = set(_STRIP.sub(" ", text.lower()).split())
    v = [1.0 if a in words else 0.0 for a in _AXES]
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v] if n else v


class _RecordingEmbedder(FunctionEmbedder):
    def __init__(self) -> None:
        super().__init__(_embed)
        self.many_batches: list[list[str]] = []

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        self.many_batches.append(list(texts))
        return [list(_embed(t)) for t in texts]


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


def _fingerprint(r) -> tuple:  # type: ignore[no-untyped-def]
    return (r.context.get("pool"), [c.claim for c in r.pool],
            [c.unit_id for c in r.pool], r.coverage, r.confidence,
            r.cache_hit, r.escalated, r.needs_retrieval)


# THE HOP GEOMETRY (Principality shape): the served hop-1 claim names the entity
# ("Kappa Gamma" — the Manic Street Preachers of this world); the hop-2 target claim
# lives in ANOTHER unit and shares no axis with the query, so only the entity probe
# "<Kappa Gamma> — <tail>" reaches it.
_HOP1_CLAIM = "The band Kappa Gamma played the alpha value show"
_BURIED_CLAIM = "Kappa Gamma released the junk three single"
_HINT_CLAIM = "Beta hardware two units"
_Q = "alpha value one"
_TAIL = "three single"


def _hop_world(**kw: object) -> tuple[SemanticCache, str]:
    """Three units; returns (cache, first-pass read_id). Tiny default budget packs
    ONLY the hop-1 group, burying the target exactly like the measured pack race."""
    ret = _WordRetriever()
    ret.set("src:a", "The band Kappa Gamma played the alpha value show.")
    ret.set("src:b", "Kappa Gamma released the junk three single.")
    ret.set("src:c", "Beta hardware two units always.")
    synth = _ClaimSynth([_HOP1_CLAIM])
    budget = kw.pop("serve_budget", 20)
    cache = _pool(ret, synth, serve_budget=budget, **kw)
    cache.get("alpha value")                       # build unit A (hop-1)
    synth.claims = [_BURIED_CLAIM]
    cache.get("junk three single")                 # build unit B (the target)
    synth.claims = [_HINT_CLAIM]
    cache.get("beta hardware two")                 # build unit C (hint terrain)
    r0 = cache.get(_Q, subs=[_TAIL])
    return cache, r0.read_id


# ----------------------------------------------------- the planted buried claim

def test_buried_claim_served_via_entity_probe() -> None:
    ev: list[dict] = []
    cache, rid = _hop_world(on_event=ev.append)
    r0_pool = cache._read_meta[rid]["served"]
    assert [t for _u, t in r0_pool] == [_HOP1_CLAIM]     # target buried at pass 1
    r1 = cache.reprobe(rid)
    assert r1 is not None
    assert r1.pool[0].claim == _BURIED_CLAIM             # entity probe rank 1
    assert _BURIED_CLAIM in r1.context["pool"]
    assert r1.parent_read_id == rid and r1.read_id != rid
    # claims-only mechanical pass: no evidence, no synthesis, no gap surface
    assert r1.evidence == [] and r1.usage is None and not r1.escalated
    assert r1.gaps == [] and r1.probe_coverage == []
    # the union probes are echoed raw-query-first, entity probe included
    assert r1.probes[0] == _Q and f"Kappa Gamma — {_TAIL}" in r1.probes
    rp = [e for e in ev if e["event"] == "reprobe"]
    assert rp and rp[-1]["entities"] == ["Kappa Gamma"]
    assert rp[-1]["parent_read_id"] == rid and rp[-1]["read_id"] == r1.read_id


# ------------------------------------------------------------ no-entities self-skip

def test_no_entities_self_skips_and_mutates_nothing() -> None:
    ev: list[dict] = []
    ret = _WordRetriever()
    ret.set("src:a", "alpha value gamma.")
    cache = _pool(ret, _ClaimSynth(["alpha value gamma"]), serve_budget=200,
                  on_event=ev.append)
    r0 = cache.get("alpha value")                  # served claims carry no proper nouns
    assert cache.reprobe(r0.read_id) is None
    skipped = [e for e in ev if e["event"] == "reprobe_skipped"]
    assert [e["reason"] for e in skipped] == ["no_entities"]
    assert [e for e in ev if e["event"] == "reprobe"] == []


def test_unknown_read_returns_none_and_unit_path_fails_loud() -> None:
    cache, _rid = _hop_world()
    assert cache.reprobe("read-999") is None
    unit_cache = SemanticCache(_WordRetriever(), _ClaimSynth(["a"]),  # type: ignore[arg-type]
                               embedder=FunctionEmbedder(_embed), read_path="unit")
    with pytest.raises(ValueError, match="reprobe requires read_path='pool'"):
        unit_cache.reprobe("read-1")


# ---------------------------------------------- union includes the originals (pin)

def test_union_includes_originals_nothing_scores_worse() -> None:
    cache, rid = _hop_world(serve_budget=400)      # room for every group
    meta = cache._read_meta[rid]
    r1 = cache.reprobe(rid)
    assert r1 is not None
    new_scores = {(c.unit_id, c.claim): c.score for c in r1.pool}
    # every originally-served claim is still served, at >= its original union score
    # (max over a SUPERSET of probes can never decrease — the displacement guard)
    served_pairs = [(u, t) for u, t in meta["served"]]
    assert served_pairs, "world must serve something at pass 1"
    for uid, text in served_pairs:
        assert (uid, text) in new_scores
    # and the buried target joined the payload on top
    assert _BURIED_CLAIM in {c.claim for c in r1.pool}


# ------------------------------------------------------------------ ONE batched embed

def test_reprobe_is_one_batched_embed() -> None:
    emb = _RecordingEmbedder()
    cache, rid = _hop_world(embedder=emb)
    meta = cache._read_meta[rid]
    emb.many_batches.clear()
    r1 = cache.reprobe(rid)
    assert r1 is not None
    assert emb.many_batches == [[
        _Q, *meta["probe_texts"], f"Kappa Gamma — {_TAIL}"]]


# ------------------------------------------------------------------ the poison guard

def test_tail_poison_guard_strips_non_initial_capitals() -> None:
    # The Madonna-cascade guard (design §two-stage; replay-gate measured): a
    # decomposer-hallucinated entity riding the sub-question tail is stripped
    # mechanically — non-initial capitalized tokens removed, trailing '?' removed —
    # before the tail pairs with a harvested entity.
    emb = _RecordingEmbedder()
    ret = _WordRetriever()
    ret.set("src:a", "The band Kappa Gamma played the alpha value show.")
    ret.set("src:b", "Kappa Gamma released the junk three single.")
    synth = _ClaimSynth([_HOP1_CLAIM])
    cache = _pool(ret, synth, serve_budget=20, embedder=emb)
    cache.get("alpha value")
    synth.claims = [_BURIED_CLAIM]
    cache.get("junk three single")
    r0 = cache.get(_Q, subs=["three Value single?"])     # 'Value' = the poison
    emb.many_batches.clear()
    r1 = cache.reprobe(r0.read_id)
    assert r1 is not None
    assert emb.many_batches == [[
        _Q, "three Value single?", "Kappa Gamma — three single"]]
    assert r1.pool[0].claim == _BURIED_CLAIM             # guarded probe still lands


# ------------------------------------------------------------------ hint extra probe

def test_hint_joins_as_one_extra_probe() -> None:
    cache, rid = _hop_world(serve_budget=400)
    plain = cache.reprobe(rid)
    assert plain is not None
    plain_scores = {c.claim: c.score for c in plain.pool}
    assert plain_scores.get(_HINT_CLAIM, 0.0) == 0.0   # unreachable without the hint
    hinted = cache.reprobe(rid, hint="beta two report")
    assert hinted is not None
    assert hinted.pool[0].claim == _HINT_CLAIM     # the hint probe hits it at 1.0
    assert "beta two report" in hinted.probes


# ------------------------------------------------- fresh read id chains + inertness

def test_reprobe_result_chains_and_repairs() -> None:
    cache, rid = _hop_world()
    r1 = cache.reprobe(rid)
    assert r1 is not None
    assert r1.read_id in cache._read_meta          # the fresh read banks its own meta
    r2 = cache.reprobe(r1.read_id)                 # chaining never crashes
    assert r2 is None or r2.parent_read_id == r1.read_id


def test_reprobe_mutates_nothing_repeat_get_byte_identical() -> None:
    cache, rid = _hop_world()
    before = _fingerprint(cache.get(_Q, subs=[_TAIL]))
    r1 = cache.reprobe(rid)
    assert r1 is not None
    after = _fingerprint(cache.get(_Q, subs=[_TAIL]))
    assert after == before                         # reprobe wrote nothing to the pool
