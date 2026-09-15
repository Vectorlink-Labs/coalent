"""v0.7b increment B — ``serve_unserved(read_id)``, the post-repair refusal rung.

ROUND-COMPOSITION-VERDICT #3 / LAB-refusal-residue recommended fix 2: on persistent
refusal, force-pack (a) the question's admitted-but-unserved REPAIRED claims (the
q053/q327 geometry: 8 claims repaired FROM the missing gold doc, repaired_served=0 —
a packing race, not discovery) and (b) the top claims of the highest-scoring
constraint-matched unit ABSENT from the served payload, then refill with the original
served claims, inside the normal budget with normal provenance. Fresh Result, new
read_id + parent link. Explicit app call, embeds-only, store never mutated.

Pinned here: the q053/q327 geometry end-to-end · the constraint-absent-unit rung on
its own · no-candidates -> None + clean skip event · never mutates the store +
repeat-get byte-identical around a call · unit-path ValueError.
"""
from __future__ import annotations

import re
from typing import Mapping

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


class _ClaimSynth:
    def __init__(self, claims: list[str]) -> None:
        self.claims = list(claims)

    def synthesize(self, query: str, chunks: list[Chunk]) -> Synthesis:
        return Synthesis(understanding={"claims": list(self.claims)},
                         used=list(range(len(chunks))))


class _WordRetriever:
    def __init__(self) -> None:
        self.docs: dict[str, tuple[str, Mapping[str, str] | None]] = {}

    def set(self, artifact_id: str, text: str,
            meta: Mapping[str, str] | None = None) -> None:
        self.docs[artifact_id] = (text, meta)

    def retrieve(self, query: str, *, namespace: str | None = None) -> list[Chunk]:
        qs = set(_STRIP.sub(" ", query.lower()).split())
        return [Chunk(artifact_id=aid, text=text, meta=meta)
                for aid, (text, meta) in self.docs.items()
                if qs & set(_STRIP.sub(" ", text.lower()).split())]


class _Extractor:
    def __init__(self, by_span: dict[str, list[str]] | None = None) -> None:
        self.by_span = dict(by_span or {})
        self.calls: list[tuple[str, str, list[str]]] = []

    def __call__(self, span: str, region: str, existing: list[str]) -> list[str]:
        self.calls.append((span, region, list(existing)))
        return list(self.by_span.get(span, []))


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


# The q053 world: unit A serves (rank 1 on the query); unit B is the "missing gold
# doc" — constraint-matched by source, but its claims never win the tiny pack.
_A_SENT = "The alpha value metric reached more units in June."
_B_SENT = "Shipments of beta hardware hit many units worldwide this quarter."
_A_META = {"title": "Alpha metric quarterly report", "source": "TechCrunch",
           "date": "2023-10-06"}
_B_META = {"title": "Beta hardware shipments dossier", "source": "Cnbc",
           "date": "2023-10-06"}
_B_GOLD = "Beta hardware shipments hit many units."      # the repaired claim


def _world(**kw: object) -> tuple[SemanticCache, str, str]:
    ret = _WordRetriever()
    ret.set("src:a", _A_SENT, meta=_A_META)
    ret.set("src:b", _B_SENT, meta=_B_META)
    synth = _ClaimSynth(["alpha value"])
    kw.setdefault("serve_budget", 10)      # packs exactly the rank-1 owner group
    cache = _pool(ret, synth, **kw)
    ra = cache.get("alpha value")
    synth.claims = ["beta kappa two"]
    rb = cache.get("beta hardware")
    assert ra.unit_id != rb.unit_id
    return cache, ra.unit_id, rb.unit_id


_CONS = {"sources": ["cnbc"]}


# ------------------------------------------- the q053/q327 geometry, end-to-end

def test_admitted_but_unserved_repaired_claims_get_packed() -> None:
    # get -> repair admits a claim INTO the constraint-matched absent unit -> pass-2
    # get still cannot pack it (it loses the ranking race, exactly q053/q327) ->
    # serve_unserved force-packs it at the head of a fresh Result.
    ev: list[dict] = []
    ex = _Extractor(by_span={_B_SENT: [_B_GOLD]})
    cache, a, b = _world(repair_extractor=ex, on_event=ev.append)
    r1 = cache.get("alpha value one", constraints=_CONS)
    assert [c.unit_id for c in r1.pool] == [a]            # B never served pass-1
    rep = cache.repair(r1.read_id)
    assert rep.admitted == 1 and rep.claims[0]["claim"] == _B_GOLD
    r2 = cache.get("alpha value one", constraints=_CONS)
    assert _B_GOLD not in r2.context["pool"]              # admitted-but-UNSERVED
    su = cache.serve_unserved(r2.read_id)
    assert su is not None
    assert su.pool[0].claim == _B_GOLD and su.pool[0].unit_id == b
    assert _B_GOLD in su.context["pool"]
    # fresh Result with the parent link + its own meta ring entry (chains compose)
    assert su.read_id and su.read_id != r2.read_id
    assert su.parent_read_id == r2.read_id
    assert su.read_id in cache._read_meta
    sus = [e for e in ev if e["event"] == "serve_unserved"]
    assert sus and sus[-1]["n_repaired"] == 1
    assert sus[-1]["parent_read_id"] == r2.read_id


def test_absent_constraint_unit_top_claims_pack_without_repair() -> None:
    # Rung (b) alone: no repair ever ran; the highest-scoring constraint-matched unit
    # ABSENT from the served payload contributes its top claims.
    ev: list[dict] = []
    cache, a, b = _world(on_event=ev.append)
    r = cache.get("alpha value one", constraints=_CONS)
    assert [c.unit_id for c in r.pool] == [a]
    su = cache.serve_unserved(r.read_id)
    assert su is not None
    assert su.pool[0].unit_id == b and su.pool[0].claim == "beta kappa two"
    sus = [e for e in ev if e["event"] == "serve_unserved"]
    assert sus[-1]["n_repaired"] == 0 and sus[-1]["n_unit_claims"] == 1
    assert sus[-1]["unserved_unit"] == b
    # the banked ledger was read NON-destructively (it stays repair's food)
    assert cache._read_bank[r.read_id]


# ------------------------------------------------------------- clean skips -> None

def test_no_candidates_returns_none_with_skip_event() -> None:
    ev: list[dict] = []
    cache, _a, _b = _world(on_event=ev.append)
    r = cache.get("alpha value one")           # no constraints, no repairs
    assert cache.serve_unserved(r.read_id) is None
    skips = [e for e in ev if e["event"] == "serve_unserved_skipped"]
    assert skips and skips[-1]["reason"] == "no_candidates"
    assert cache.serve_unserved("read-999") is None
    assert [e for e in ev if e["event"] == "serve_unserved_skipped"][-1]["reason"] \
        == "unknown_read"


def test_served_repaired_claim_is_not_reforced() -> None:
    # With a budget big enough that the repaired claim DID serve at pass-2, there is
    # nothing unserved left — the rung self-skips instead of re-packing noise.
    ev: list[dict] = []
    ex = _Extractor(by_span={_B_SENT: [_B_GOLD]})
    cache, _a, _b = _world(serve_budget=200, repair_extractor=ex, on_event=ev.append)
    r1 = cache.get("alpha value one", constraints=_CONS)
    assert cache.repair(r1.read_id).admitted == 1
    r2 = cache.get("alpha value one", constraints=_CONS)
    assert _B_GOLD in r2.context["pool"]                  # it served this time
    assert cache.serve_unserved(r2.read_id) is None
    assert [e for e in ev if e["event"] == "serve_unserved_skipped"][-1]["reason"] \
        == "no_candidates"


# --------------------------------------------------- store never mutated + inert

def test_never_mutates_store_and_repeat_get_byte_identical() -> None:
    ex = _Extractor(by_span={_B_SENT: [_B_GOLD]})
    cache, _a, _b = _world(repair_extractor=ex)
    r1 = cache.get("alpha value one", constraints=_CONS)
    cache.repair(r1.read_id)
    r2 = cache.get("alpha value one", constraints=_CONS)
    g_before = cache.get("alpha value one")
    snap = {uid: (list(u.understanding.get("claims") or []),
                  len(u.claim_embeddings or ()), u.status)
            for uid, u in cache._units.items()}
    epoch = cache._rows_epoch
    assert cache.serve_unserved(r2.read_id) is not None
    assert cache._rows_epoch == epoch                     # no reindex, no commit
    assert {uid: (list(u.understanding.get("claims") or []),
                  len(u.claim_embeddings or ()), u.status)
            for uid, u in cache._units.items()} == snap
    assert _fingerprint(cache.get("alpha value one")) == _fingerprint(g_before)


# ------------------------------------------------------------------ arming contract

def test_unit_path_fails_loud() -> None:
    ret = _WordRetriever()
    ret.set("src:a", _A_SENT)
    unit_cache = SemanticCache(ret, _ClaimSynth(["alpha value"]),     # type: ignore[arg-type]
                               embedder=FunctionEmbedder(_embed),
                               hit_threshold=0.5, coverage_floor=0.0)
    with pytest.raises(ValueError, match="requires read_path='pool'"):
        unit_cache.serve_unserved("read-1")
