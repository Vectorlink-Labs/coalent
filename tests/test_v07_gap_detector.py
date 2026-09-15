"""v0.7 Phase B increment 2 — the gap detector as a SIGNAL (opt-in ``gap_detector``).

PIPELINE-DESIGN-v07 §THE STACK item 3 / lab arm C: per-probe best evidence-sentence
cosine vs best claim-pool cosine; a span beating the claims by the module-constant margin
(_GAP_DELTA = 0.02, deliberately not a tunable) is a measured EXTRACTION HOLE (the source
has the fact, the claim tier doesn't — repair terrain, banked as a candidate); both tiers
weak is a CORPUS HOLE (nothing in store — tool-routing terrain). The round's central
discovery is why this ships: confident-wrong first passes fire NOTHING refusal-gated;
the detector is the one signal that fires WITHOUT a refusal.

SPEC DELTA under test (residual-density sweep 2026-09-04): the span source is the FULL
evidence-sentence tier, hydrated LAZILY per unit (split + ONE batched embed, side store
keyed unit id + evidence content hash — never at ingest), NOT the residual tier (measured
dead at residual density: 0/130 fires survive). The residual machinery stays untouched.

The safety property pinned hardest here: the detector OBSERVES, NEVER PACKS — serving is
byte-identical with the knob off (default) AND with the knob on.
"""
from __future__ import annotations

import re

import pytest

from coalent import FunctionEmbedder
from coalent.semantic import Chunk, ResidualSpan, SemanticCache, Synthesis
from coalent.semantic.cache import _GAP_DELTA, _READ_LOG_CAP

_AXES = ("alpha", "beta", "gamma", "kappa", "one", "two", "three", "value", "junk")
_STRIP = re.compile(r"[^\w\s]")


def _embed(text: str) -> list[float]:
    words = set(_STRIP.sub(" ", text.lower()).split())
    v = [1.0 if a in words else 0.0 for a in _AXES]
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v] if n else v


class _RecordingEmbedder(FunctionEmbedder):
    """Records every ``embed_many`` batch — the hydration-laziness instrument."""

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


# One news-shaped doc: the claim covers only the alpha sentence; the beta sentence is the
# PLANTED EXTRACTION HOLE (it exists in evidence, no claim carries it).
_ALPHA_SENT = "The alpha value metric reached more units in June."
_BETA_SENT = "Shipments of beta hardware hit many units worldwide this quarter."
_DOC = _ALPHA_SENT + " " + _BETA_SENT


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


_BETA_SUBS = {"alpha value one": [{"q": "beta kappa", "hyde": None}]}


def _beta_decomp(q: str) -> list[dict]:
    return _BETA_SUBS.get(q, [])


# ------------------------------------------------------------------ constructor contract

def test_gap_detector_constructor_contract() -> None:
    ret = _WordRetriever()
    synth = _ClaimSynth(["alpha"])
    # structurally inert on the unit path -> fail LOUD (query_keys/decompose precedent)
    with pytest.raises(ValueError, match="gap_detector requires read_path='pool'"):
        SemanticCache(ret, synth, embedder=FunctionEmbedder(_embed),  # type: ignore[arg-type]
                      gap_detector=True)
    # SPEC DELTA pin: residual_spans is NOT required — the detector hydrates its own
    # evidence-sentence tier (the residual tier was measured dead as its span source).
    _pool(ret, synth, gap_detector=True)
    _pool(ret, synth, gap_detector=True, residual_spans=False)


# ------------------------------------------------------------- default-OFF byte-inert

def test_default_off_byte_inert_and_fields_empty() -> None:
    cache_absent = _hole_world()
    cache_false = _hole_world(gap_detector=False)
    for q in ("alpha value one", "beta kappa", "junk two"):
        ra, rf = cache_absent.get(q), cache_false.get(q)
        assert _fingerprint(ra) == _fingerprint(rf)
        for r in (ra, rf):                       # inert by construction when off
            assert r.probes == [] and r.probe_coverage == [] and r.gaps == []


# ---------------------------------------------------- observes, never packs (knob ON)

def test_knob_on_serving_byte_identical() -> None:
    # THE increment bar: with the knob ON, serving — payload bytes, served claims,
    # coverage, gate outcome, escalation — is byte-identical to OFF; only the new
    # observational fields differ. Decomposed and plain reads both pinned.
    cache_off = _hole_world(decompose=_beta_decomp)
    cache_on = _hole_world(decompose=_beta_decomp, gap_detector=True)
    for q in ("alpha value one", "beta kappa", "alpha gamma", "junk two"):
        r_off, r_on = cache_off.get(q), cache_on.get(q)
        assert _fingerprint(r_on) == _fingerprint(r_off)
        assert r_off.probes == [] and r_on.probes[0] == q   # raw query always first
        assert r_on.probe_coverage and all(
            set(row) == {"probe", "best_claim", "best_span", "margin", "fired"}
            for row in r_on.probe_coverage)


# ------------------------------------------------------- the planted extraction hole

def test_planted_extraction_hole_fires_and_banks() -> None:
    ev: list[dict] = []
    cache = _hole_world(decompose=_beta_decomp, gap_detector=True, on_event=ev.append)
    r = cache.get("alpha value one")
    # the probe union scored [raw query, "beta kappa"]
    assert r.probes == ["alpha value one", "beta kappa"]
    by_probe = {row["probe"]: row for row in r.probe_coverage}
    # the beta probe: no claim carries beta (best_claim 0), the evidence sentence does —
    # margin clears the module-constant delta and fires
    beta = by_probe["beta kappa"]
    assert beta["best_claim"] == 0.0 and beta["best_span"] > 0.5
    assert beta["fired"] and beta["margin"] > _GAP_DELTA
    holes = [g for g in r.gaps if g["kind"] == "extraction_hole"]
    assert [g["probe"] for g in holes] == ["beta kappa"]
    assert holes[0]["span"] == _BETA_SENT and holes[0]["source"] == "src:news"
    assert holes[0]["unit_id"] == r.pool[0].unit_id
    # the fired span is BANKED as a repair candidate on the read's ledger
    bank = cache._read_bank[r.read_id]
    assert [(b["origin"], b["kind"], b["span"]) for b in bank] \
        == [("detector", "extraction_hole", _BETA_SENT)]
    fired_events = [e for e in ev if e["event"] == "gap_detector"]
    assert fired_events[-1]["fired"] == 1 and fired_events[-1]["banked"] == 1


def test_covered_probe_does_not_fire() -> None:
    cache = _hole_world(gap_detector=True,
                        decompose=lambda q: [{"q": "alpha value", "hyde": None}]
                        if q == "alpha value one" else [])
    r = cache.get("alpha value one")
    by_probe = {row["probe"]: row for row in r.probe_coverage}
    covered = by_probe["alpha value"]
    # the claim tier reaches the probe exactly — the span can't beat it by the margin
    assert covered["best_claim"] == pytest.approx(1.0)
    assert not covered["fired"]
    assert not [g for g in r.gaps if g["probe"] == "alpha value"]


def test_corpus_hole_classification() -> None:
    # a probe NEITHER tier reaches (both cosines ~0, under span_serve_floor): the corpus
    # doesn't contain this sub-answer -> corpus_hole, routed to a tool node — and NOT
    # banked (there is nothing to repair from).
    cache = _hole_world(gap_detector=True,
                        decompose=lambda q: [{"q": "junk", "hyde": None}]
                        if q == "alpha value one" else [])
    r = cache.get("alpha value one")
    holes = [g for g in r.gaps if g["probe"] == "junk"]
    assert [g["kind"] for g in holes] == ["corpus_hole"]
    assert all(b["kind"] != "corpus_hole" for b in cache._read_bank.get(r.read_id, []))


def test_q_only_degraded_mode() -> None:
    # No decompose/subs armed: the detector runs on the raw query alone (allowed but
    # documented — q-only under-fires in the measured evidence; the knob still works).
    cache = _hole_world(gap_detector=True)
    r = cache.get("beta kappa")
    assert r.probes == ["beta kappa"]
    assert len(r.probe_coverage) == 1


# ------------------------------------------------- lazy hydration + the warm-cache seam

def test_sentence_hydration_is_lazy_and_cached() -> None:
    emb = _RecordingEmbedder()
    ret = _WordRetriever()
    ret.set("src:news", _DOC)
    cache = _pool(ret, _ClaimSynth(["alpha value"]), embedder=emb,
                  serve_budget=200, gap_detector=True)
    cache.get("alpha value")                     # build + first detector use: hydrates
    sent_batches = [b for b in emb.many_batches if _BETA_SENT in b]
    assert len(sent_batches) == 1                # ONE batched embed for the unit's tier
    assert _ALPHA_SENT in sent_batches[0]        # FULL tier: covered sentences included
    emb.many_batches.clear()
    cache.get("alpha value one")                 # warm side store: nothing re-embeds
    assert [b for b in emb.many_batches if _BETA_SENT in b] == []


def test_side_store_is_the_warm_cache_seam() -> None:
    # A driver holding precomputed sentence embeddings (e.g. the bench's mech5-spans
    # caches) may pre-seed the side store keyed (unit id, evidence content hash): the
    # detector then embeds NOTHING and scores the seeded rows.
    emb = _RecordingEmbedder()
    ret = _WordRetriever()
    ret.set("src:a", "The alpha value gamma metric reached more units.")
    cache = _pool(ret, _ClaimSynth(["alpha value"]), embedder=emb,
                  serve_budget=200, gap_detector=True,
                  decompose=lambda q: [{"q": "beta kappa", "hyde": None}]
                  if q == "alpha value one" else [])
    r0 = cache.get("alpha value")                # build (detector hydrates the real tier)
    uid = r0.pool[0].unit_id
    unit = cache._units[uid]
    seeded = "Seeded beta kappa sentence from the warm cache store."
    cache._gap_sent_cache[uid] = (cache._gap_content_key(unit), (
        ResidualSpan(text=seeded, artifact_id="src:seed", chunk_idx=0,
                     embedding=tuple(_embed(seeded))),))
    cache._gap_state = None                      # driver seam: invalidate the ns view
    emb.many_batches.clear()
    r = cache.get("alpha value one")
    assert emb.many_batches == [[  # the read's ONE query+probe batch — no sentence embeds
        "alpha value one", "beta kappa"]]
    holes = [g for g in r.gaps if g["kind"] == "extraction_hole"]
    assert [g["span"] for g in holes] == [seeded]
    assert holes[0]["source"] == "src:seed"      # scored the SEEDED rows, provenance kept


def test_rebuild_rehydrates_only_the_changed_unit() -> None:
    emb = _RecordingEmbedder()
    ret = _WordRetriever()
    b_sent = "Some gamma committee sentence with enough characters here."
    ret.set("src:news", _DOC)
    ret.set("src:b", b_sent)
    cache = _pool(ret, _ClaimSynth(["alpha value"]), embedder=emb,
                  serve_budget=200, gap_detector=True)
    cache.get("alpha value")                     # unit A
    cache.get("gamma committee")                 # unit B (claims replaced per synth)
    emb.many_batches.clear()
    new_beta = "Entirely new beta sentence appended to the changed source."
    ret.set("src:news", _DOC + " " + new_beta)
    cache.source_changed("src:news", text=_DOC + " " + new_beta)
    cache.get("alpha value")                     # rebuild A in place; B stays warm
    rehydrated = [b for b in emb.many_batches if new_beta in b]
    assert len(rehydrated) == 1                  # exactly ONE batch re-embeds A's tier
    assert all(b_sent not in b for b in emb.many_batches)   # B never re-embeds


# ------------------------------------------------------------------ ledger ring cap

def test_ledger_ring_evicts_oldest_read() -> None:
    cache = _hole_world(decompose=lambda q: [{"q": "beta kappa", "hyde": None}],
                        gap_detector=True)
    ids: list[str] = []
    for _ in range(_READ_LOG_CAP + 3):
        ids.append(cache.get("alpha value one").read_id)
    assert len(cache._read_bank) == _READ_LOG_CAP
    assert ids[0] not in cache._read_bank        # oldest evicted
    assert ids[-1] in cache._read_bank
