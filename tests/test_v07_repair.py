"""v0.7 Phase B increment 4 — ``repair(read_id)``, the pump (opt-in ``repair_extractor``).

PIPELINE-DESIGN-v07 §THE STACK item 7 / lab arm F — the only mechanism with grounded
end-to-end wins: a read's candidate SPANS (banked detector fires + constraints matches,
plus bridge candidates derived at repair time from the read's served claims) go through
the BYO span-anchored extractor and the mechanical dedup (exact-norm OR cos >= 0.95 vs
the unit's claims); survivors append to the owning unit — claims + inline embeddings +
per-claim provenance — append-only, capped, persisted. The repaired claim then serves
at the NEXT get() through the UNCHANGED ranker (measured rank 1 on GameStop).

Pinned here: next-get serving · zero-repeat dedup (both bars) · the span-anchored call
shape (span + bounded ±1-sentence region + do-not-repeat list) · the per-read admission
cap · clean no-op on unknown reads · constraints-only feeding with the detector off ·
serde round-trip of repaired units · read-path byte-inertness of the arm.
"""
from __future__ import annotations

import re
from typing import Mapping

import pytest

from coalent import FunctionEmbedder
from coalent.semantic import (
    Chunk,
    RepairReport,
    SemanticCache,
    SQLiteCognitionStore,
    Synthesis,
)
from coalent.semantic.cache import _REPAIR_CAP, _REPAIR_VIA

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
    """Recording fake extractor: claims keyed by exact span text; every call shape
    (span, region, do-not-repeat list) is captured for the span-anchor pin."""

    def __init__(self, by_span: dict[str, list[str]] | None = None,
                 default: list[str] | None = None) -> None:
        self.by_span = dict(by_span or {})
        self.default = list(default or [])
        self.calls: list[tuple[str, str, list[str]]] = []

    def __call__(self, span: str, region: str, existing: list[str]) -> list[str]:
        self.calls.append((span, region, list(existing)))
        return list(self.by_span.get(span, self.default))


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


# One news-shaped doc: the claim covers only the alpha sentence; the beta sentence is
# the PLANTED EXTRACTION HOLE (in evidence, no claim carries it) — the detector world
# of the increment-2 suite, reused verbatim.
_ALPHA_SENT = "The alpha value metric reached more units in June."
_BETA_SENT = "Shipments of beta hardware hit many units worldwide this quarter."
_DOC = _ALPHA_SENT + " " + _BETA_SENT
_BETA_CLAIM = "Beta hardware shipments hit many units."   # tokens -> beta axis only


def _beta_decomp(q: str) -> list[dict]:
    return [{"q": "beta kappa", "hyde": None}] if q == "alpha value one" else []


def _hole_world(**kw: object) -> SemanticCache:
    ret = _WordRetriever()
    ret.set("src:news", _DOC)
    synth = _ClaimSynth(["alpha value"])
    kw.setdefault("serve_budget", 200)
    cache = _pool(ret, synth, **kw)
    cache.get("alpha value")                              # build the one unit
    return cache


def _fingerprint(r) -> tuple:  # type: ignore[no-untyped-def]
    return (r.context.get("pool"), [c.claim for c in r.pool],
            [c.unit_id for c in r.pool], r.coverage, r.confidence,
            r.cache_hit, r.escalated, r.needs_retrieval)


# ------------------------------------------------------------------ arming contract

def test_repair_constructor_and_arming_contract() -> None:
    ret = _WordRetriever()
    synth = _ClaimSynth(["alpha"])
    # structurally inert on the unit path -> fail LOUD (gap_detector precedent)
    with pytest.raises(ValueError, match="repair_extractor requires read_path='pool'"):
        SemanticCache(ret, synth, embedder=FunctionEmbedder(_embed),  # type: ignore[arg-type]
                      repair_extractor=_Extractor(), read_path="unit")
    # a non-callable arm is a contract error (the library never calls an LLM)
    with pytest.raises(TypeError, match="repair_extractor must be None or a callable"):
        SemanticCache(ret, synth, embedder=FunctionEmbedder(_embed),  # type: ignore[arg-type]
                      read_path="pool", pool_header=_header,
                      repair_extractor=True)                          # type: ignore[arg-type]
    # calling repair() without the callable raises the clear arming error
    cache = _hole_world()
    r = cache.get("alpha value one")
    with pytest.raises(RuntimeError, match="repair_extractor"):
        cache.repair(r.read_id)


# ------------------------------------------- the pump: serve on the NEXT get()

def test_repaired_claim_serves_on_next_get() -> None:
    ev: list[dict] = []
    ex = _Extractor(by_span={_BETA_SENT: [_BETA_CLAIM]})
    cache = _hole_world(decompose=_beta_decomp, gap_detector=True,
                        repair_extractor=ex, on_event=ev.append)
    r0 = cache.get("alpha value one")
    assert [g["kind"] for g in r0.gaps] == ["extraction_hole"]   # the detector banked
    report = cache.repair(r0.read_id)
    assert isinstance(report, RepairReport)
    assert report.read_id == r0.read_id
    assert report.candidates_seen >= 1 and report.admitted == 1
    prov = report.claims[0]
    assert prov["claim"] == _BETA_CLAIM and prov["span"] == _BETA_SENT
    assert prov["source"] == "src:news" and prov["via"] == _REPAIR_VIA
    assert prov["origin"] == "detector" and prov["ts"] > 0
    assert report.units_touched == [r0.pool[0].unit_id]
    applied = [e for e in ev if e["event"] == "repair_applied"]
    assert [(e["unit_id"], e["claims_added"]) for e in applied] \
        == [(r0.pool[0].unit_id, 1)]
    # the NEXT get() serves the repaired claim through the UNCHANGED ranker: the beta
    # probe now hits a real pool row at rank 1 (cosine 1.0 on the beta axis)
    r1 = cache.get("beta kappa")
    assert r1.pool and r1.pool[0].claim == _BETA_CLAIM
    assert _BETA_CLAIM in r1.context["pool"]
    # the unit carries the appended claim + inline embedding + provenance record
    unit = cache._units[r0.pool[0].unit_id]
    assert unit.understanding["claims"] == ["alpha value", _BETA_CLAIM]
    assert len(unit.claim_embeddings) >= len(unit.understanding["claims"])
    assert unit.understanding["_repair_provenance"][0]["claim"] == _BETA_CLAIM
    # the banked ledger was CONSUMED
    assert r0.read_id not in cache._read_bank


# ------------------------------------------------------------------ mechanical dedup

def test_dedup_admits_zero_repeats_exact_and_cosine() -> None:
    # exact-norm repeat (case/whitespace variant) AND a 0.95-cosine rephrasing (same
    # token axes -> cosine 1.0) must BOTH be rejected; nothing is appended.
    ex = _Extractor(default=["  ALPHA   value ", "Alpha metric of value"])
    cache = _hole_world(decompose=_beta_decomp, gap_detector=True,
                        repair_extractor=ex)
    r0 = cache.get("alpha value one")
    uid = r0.pool[0].unit_id
    report = cache.repair(r0.read_id)
    assert report.extracted >= 2 and report.admitted == 0
    assert report.claims == [] and report.units_touched == []
    assert cache._units[uid].understanding["claims"] == ["alpha value"]
    assert "_repair_provenance" not in cache._units[uid].understanding


# ------------------------------------------------------- span-anchored call shape

def test_extractor_receives_span_region_and_do_not_repeat() -> None:
    ex = _Extractor(by_span={_BETA_SENT: [_BETA_CLAIM]})
    cache = _hole_world(decompose=_beta_decomp, gap_detector=True,
                        repair_extractor=ex)
    r0 = cache.get("alpha value one")
    cache.repair(r0.read_id)
    beta_calls = [c for c in ex.calls if c[0] == _BETA_SENT]
    assert beta_calls, "the banked span must anchor its own extractor call"
    span, region, existing = beta_calls[0]
    assert span == _BETA_SENT
    # bounded ±1-sentence region: the neighbor sentence, never the span alone
    assert _ALPHA_SENT in region and _BETA_SENT in region
    # the unit's current claims are the do-not-repeat list
    assert existing == ["alpha value"]


# ------------------------------------------------------------------ per-read cap

def test_per_read_admission_cap_stops_extracting() -> None:
    # capitalized starts: the v0.7b admission hygiene rejects lowercase-start claims
    many = ["Beta one", "Beta two", "Beta three", "Kappa one", "Kappa two",
            "Kappa three", "Gamma one", "Gamma two", "Gamma three", "Junk one"]
    ex = _Extractor(default=many)
    cache = _hole_world(decompose=_beta_decomp, gap_detector=True,
                        repair_extractor=ex)
    r0 = cache.get("alpha value one")
    report = cache.repair(r0.read_id)
    assert report.admitted == _REPAIR_CAP
    assert len(report.claims) == _REPAIR_CAP
    # the cap also stops further extractor calls (bridge candidates never re-pay)
    assert len(ex.calls) == 1
    unit = cache._units[r0.pool[0].unit_id]
    assert len(unit.understanding["claims"]) == 1 + _REPAIR_CAP


# ------------------------------------------------------------------ clean no-ops

def test_unknown_read_is_clean_noop() -> None:
    ex = _Extractor(default=[_BETA_CLAIM])
    cache = _hole_world(repair_extractor=ex)
    report = cache.repair("read-999")
    assert report == RepairReport(read_id="read-999")
    assert ex.calls == []


def test_no_candidates_is_clean_noop_report() -> None:
    # a read that served NOTHING (empty pool namespace) banks nothing and seeds no
    # bridge — repair is a zero-candidate no-op, no extractor call, nothing raised.
    ret = _WordRetriever()
    cache = _pool(ret, _ClaimSynth(["alpha value"]), serve_budget=200,
                  repair_extractor=(ex := _Extractor(default=[_BETA_CLAIM])))
    r = cache.get("junk two", namespace="empty")
    report = cache.repair(r.read_id)
    assert report.candidates_seen == 0 and report.admitted == 0
    assert ex.calls == []


# ------------------------------------- constraints feed with the detector OFF

def test_repair_from_constraints_candidates_detector_off() -> None:
    ret = _WordRetriever()
    ret.set("src:news", _DOC,
            meta={"title": "Beta quarter", "source": "Daily Alpha",
                  "date": "2023-11-02"})
    ex = _Extractor(by_span={_BETA_SENT: [_BETA_CLAIM]})
    cache = _pool(ret, _ClaimSynth(["alpha value"]), serve_budget=200,
                  repair_extractor=ex)          # gap_detector NOT armed
    cache.get("alpha value")                    # build (captures source_meta)
    r = cache.get("alpha value one", constraints={"dates": ["November 2, 2023"]})
    assert r.gaps == []                          # detector off: no gap surface at all
    banked = cache._read_bank[r.read_id]
    assert {b["origin"] for b in banked} == {"constraints"}
    report = cache.repair(r.read_id)
    assert report.admitted == 1 and report.claims[0]["origin"] == "constraints"
    assert cache.get("beta kappa").pool[0].claim == _BETA_CLAIM


# ------------------------------------------------------- bridge feed at repair time

def test_bridge_candidates_derived_from_served_claims() -> None:
    # NO detector, NO constraints: the read's served claim ("alpha value") is the
    # bridge seed; its vector reaches the alpha evidence sentence, so the sentence
    # tier's top spans join the queue at repair time — the arm-D feeder path.
    ex = _Extractor(by_span={_BETA_SENT: [_BETA_CLAIM]})
    cache = _hole_world(repair_extractor=ex)
    r = cache.get("alpha value one")
    assert cache._read_bank.get(r.read_id) is None      # nothing banked at read time
    report = cache.repair(r.read_id)
    assert report.candidates_seen >= 1                  # bridge-derived only
    assert all(c[0] in (_ALPHA_SENT, _BETA_SENT) for c in ex.calls)
    # the beta span is in the tier (2 spans total, bridge_k=3) -> its claim admits
    assert report.admitted == 1 and report.claims[0]["origin"] == "bridge"
    assert cache.get("beta kappa").pool[0].claim == _BETA_CLAIM


# ---------------------------------------------------- queue is relevance-ordered

def test_queue_consumed_best_score_first() -> None:
    # The NEWS-02 replay-gate starvation pin: a LOW-score candidate banked first
    # must not eat the admission cap before a higher-score candidate is extracted —
    # the queue is consumed best banked score first ("the matched units' BEST spans
    # join the queue"), stable on ties.
    many8 = ["Beta one", "Beta two", "Beta three", "Kappa one",
             "Kappa two", "Kappa three", "Gamma one", "Gamma two"]
    ex = _Extractor(default=many8)
    cache = _hole_world(repair_extractor=ex)
    uid = cache.get("alpha value one").pool[0].unit_id
    r = cache.get("junk two", namespace="empty")     # served nothing: no bridge feed
    cache._bank_candidates(r.read_id, [
        {"probe": "", "span": _ALPHA_SENT, "unit_id": uid, "source": "src:news",
         "kind": "metadata", "score": 0.2, "origin": "constraints"},
        {"probe": "", "span": _BETA_SENT, "unit_id": uid, "source": "src:news",
         "kind": "extraction_hole", "score": 0.9, "origin": "detector"},
    ])
    report = cache.repair(r.read_id)
    assert report.admitted == _REPAIR_CAP
    assert [c[0] for c in ex.calls] == [_BETA_SENT]   # best-first; cap stops the rest
    assert {c["origin"] for c in report.claims} == {"detector"}


def test_banked_candidates_consume_before_bridge() -> None:
    # The NEWS-02/-03 cross-metric pin: bridge scores (seed-claim -> span cosines,
    # systematically higher) must never outrank the read's own banked candidates —
    # the bridge APPENDS after the ledger, scores never compared across metrics.
    many8 = ["Beta one", "Beta two", "Beta three", "Kappa one",
             "Kappa two", "Kappa three", "Gamma one", "Gamma two"]
    ex = _Extractor(default=many8)
    cache = _hole_world(repair_extractor=ex)
    r = cache.get("alpha value one")         # serves the alpha claim -> bridge seeds
    uid = r.pool[0].unit_id
    cache._bank_candidates(r.read_id, [      # low-score banked candidate (0.1) vs the
        {"probe": "", "span": _BETA_SENT, "unit_id": uid, "source": "src:news",
         "kind": "metadata", "score": 0.1, "origin": "constraints"}])
    report = cache.repair(r.read_id)         # bridge alpha span cos ~1.0 exists
    assert report.admitted == _REPAIR_CAP
    assert [c[0] for c in ex.calls] == [_BETA_SENT]   # banked first, bridge after
    assert {c["origin"] for c in report.claims} == {"constraints"}


# ------------------------------------------------------------------ error contract

def test_raising_extractor_skips_candidate_not_call() -> None:
    class _Flaky(_Extractor):
        def __call__(self, span: str, region: str, existing: list[str]) -> list[str]:
            super().__call__(span, region, existing)
            if span == _ALPHA_SENT:
                raise RuntimeError("boom")
            return [_BETA_CLAIM] if span == _BETA_SENT else []

    ex = _Flaky()
    cache = _hole_world(repair_extractor=ex)
    r = cache.get("alpha value one")
    report = cache.repair(r.read_id)                     # never raises
    assert report.admitted == 1 and report.claims[0]["claim"] == _BETA_CLAIM


# ------------------------------------------ admission hygiene (v0.7b, LAB-displacement)

# THE two banked defect texts, verbatim (LAB-displacement §5: both served at pack-head
# ranks 1-3 in the displacement dissection — q346 / q089).
_DEFECT_PRONOUN = "He was the richest person in the world under 30"
_DEFECT_TRUNC = "The Sporting News provided updates and highlights from Jaguars vs."


def test_hygiene_rejects_banked_defect_texts_verbatim() -> None:
    ex = _Extractor(by_span={_BETA_SENT: [_DEFECT_PRONOUN, _DEFECT_TRUNC]})
    cache = _hole_world(decompose=_beta_decomp, gap_detector=True,
                        repair_extractor=ex)
    r0 = cache.get("alpha value one")
    uid = r0.pool[0].unit_id
    report = cache.repair(r0.read_id)
    assert report.extracted >= 2
    assert report.rejected == 2                       # both defects counted
    assert report.admitted == 0 and report.claims == []
    assert cache._units[uid].understanding["claims"] == ["alpha value"]


def test_hygiene_defect_rules_mechanical() -> None:
    d = SemanticCache._claim_defect
    # antecedent-free pronoun subjects (leading he/she/it/they/this/that, no proper noun)
    assert d(_DEFECT_PRONOUN) == "pronoun_subject"
    assert d("They reported a loss for the quarter") == "pronoun_subject"
    assert d("This is potentially dangerous") == "pronoun_subject"
    # a pronoun subject WITH a proper noun referent stays admissible
    assert d("He joined Michigan for a trip to State College") is None
    assert d("It cited FTX in the court filing") is None
    # truncations: leading ellipsis / lowercase start / dangling connector or punct tail
    assert d(_DEFECT_TRUNC) == "truncated"
    assert d("...updates and highlights from the game") == "truncated"
    assert d("updates and highlights from the game") == "truncated"
    assert d("The Warriors have two remaining lottery picks,") == "truncated"
    assert d("Beta hardware shipments hit many units.") is None
    assert d("30 percent of the units shipped in June") is None   # digit start is fine


def test_hygiene_mixed_call_admits_good_counts_rejected() -> None:
    ex = _Extractor(by_span={_BETA_SENT: [_DEFECT_PRONOUN, _BETA_CLAIM, _DEFECT_TRUNC]})
    cache = _hole_world(decompose=_beta_decomp, gap_detector=True,
                        repair_extractor=ex)
    r0 = cache.get("alpha value one")
    report = cache.repair(r0.read_id)
    assert report.rejected == 2 and report.admitted == 1
    assert report.claims[0]["claim"] == _BETA_CLAIM
    assert cache.get("beta kappa").pool[0].claim == _BETA_CLAIM


# ------------------------------------------------------------------ serde round-trip

def test_repaired_unit_persists_and_reloads(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = SQLiteCognitionStore(str(tmp_path / "repair.db"))
    ret = _WordRetriever()
    ret.set("src:news", _DOC)
    synth = _ClaimSynth(["alpha value"])
    ex = _Extractor(by_span={_BETA_SENT: [_BETA_CLAIM]})
    cache = _pool(ret, synth, serve_budget=200, store=store,
                  decompose=_beta_decomp, gap_detector=True, repair_extractor=ex)
    cache.get("alpha value")
    r0 = cache.get("alpha value one")
    assert cache.repair(r0.read_id).admitted == 1
    # a FRESH cache over the same store (full JSON serde round-trip via SQLite)
    reloaded = _pool(ret, synth, serve_budget=200, store=store)
    r1 = reloaded.get("beta kappa")
    assert r1.pool and r1.pool[0].claim == _BETA_CLAIM
    unit = reloaded._units[r0.pool[0].unit_id]
    assert unit.understanding["_repair_provenance"][0]["via"] == _REPAIR_VIA
    assert len(unit.claim_embeddings) >= len(unit.understanding["claims"])


# ------------------------------------------------------------------ read byte-inert

def test_armed_extractor_never_changes_serving() -> None:
    cache_off = _hole_world(decompose=_beta_decomp)
    cache_on = _hole_world(decompose=_beta_decomp,
                           repair_extractor=_Extractor(default=[_BETA_CLAIM]))
    for q in ("alpha value one", "beta kappa", "junk two"):
        assert _fingerprint(cache_on.get(q)) == _fingerprint(cache_off.get(q))
