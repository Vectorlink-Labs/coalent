"""v0.7 Phase B increment 3 — the upstream params: ``subs=`` and ``constraints=`` on get().

PIPELINE-DESIGN-v07 §THE AGENTIC CONTRACT items 1-2 / §UPSTREAM:

``subs=`` — PLANNER-OWNED decomposition. The planner's grounded sub-questions WIN over
the blind BYO HyDE callable (the Madonna hyde-cascade is why); the callable stays as the
fallback for naked deployments. Both routes feed the SAME probe-union normalization, so
identical content serves BYTE-IDENTICALLY (the increment's pin).

``constraints=`` — {dates, sources, entities} vs unit ingest metadata (source_meta:
dates through the ONE ISO canonicalizer, sources/entities case-insensitive substring vs
source/title). FEEDER ONLY: matched units' best evidence spans bank into the read's
repair-candidate ledger; pool scoring and serving are untouched EVER (pinned — the BM25
pool-scoring precedent is why the restraint is structural). No match = clean no-op.
"""
from __future__ import annotations

import logging
import re
from typing import Mapping

import pytest

from coalent import FunctionEmbedder
from coalent.semantic import Chunk, SemanticCache, Synthesis
from coalent.semantic.cache import _SPANS_PER_READ, _canonical_date

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
    """The decompose suite's burial world: tiny budget packs ONE owner group, so
    whichever claim ranks first is the whole payload — probe effects are byte-visible."""
    ret = _WordRetriever()
    ret.set("src:a", "alpha value gamma.")
    ret.set("src:b", "beta kappa two.")
    synth = _ClaimSynth(["alpha value"])
    kw.setdefault("serve_budget", 10)
    cache = _pool(ret, synth, **kw)
    ra = cache.get("alpha value")
    synth.claims = ["beta kappa two"]
    rb = cache.get("beta kappa")
    assert ra.unit_id != rb.unit_id
    return cache, ra.unit_id, rb.unit_id


def _fingerprint(r) -> tuple:  # type: ignore[no-untyped-def]
    return (r.context["pool"], [c.claim for c in r.pool],
            [c.unit_id for c in r.pool], r.coverage, r.confidence,
            r.cache_hit, r.escalated, r.needs_retrieval)


_SUB = {"q": "beta kappa two", "hyde": None}


def _callable_world(**kw: object) -> tuple[SemanticCache, str, str]:
    return _two_unit_world(
        decompose=lambda q: [dict(_SUB)] if q == "alpha value one" else [], **kw)


# --------------------------------------------------------------------- subs= ownership

def test_subs_byte_identical_to_callable_decomposition() -> None:
    # THE increment pin: identical sub-question content through subs= serves a payload
    # byte-identical to the callable route — same union, same clamp, same everything.
    cache_fn, _, _ = _callable_world()
    cache_subs, _, _ = _two_unit_world()
    r_fn = cache_fn.get("alpha value one")
    r_subs = cache_subs.get("alpha value one", subs=[dict(_SUB)])
    assert _fingerprint(r_subs) == _fingerprint(r_fn)
    assert "beta kappa two" in r_subs.context["pool"]     # the probe did the lifting
    # bare-string form is the same read
    r_str = cache_subs.get("alpha value one", subs=["beta kappa two"])
    assert _fingerprint(r_str) == _fingerprint(r_fn)


def test_subs_win_over_callable() -> None:
    calls: list[str] = []

    def decomp(q: str) -> list[dict]:
        calls.append(q)
        return [{"q": "junk", "hyde": None}]              # would find nothing

    cache, _, b = _two_unit_world(decompose=decomp)
    calls.clear()
    r = cache.get("alpha value one", subs=[dict(_SUB)])   # planner content wins
    assert calls == []                                    # callable NOT consulted
    assert r.pool[0].claim == "beta kappa two" and r.pool[0].unit_id == b
    # subs=[] is the sanctioned EXPLICIT no-decomposition — still no callable call
    cache_off, _, _ = _two_unit_world()
    r_empty = cache.get("alpha value one", subs=[])
    assert calls == []
    assert _fingerprint(r_empty) == _fingerprint(cache_off.get("alpha value one"))
    # subs=None falls back to the callable (the naked-deployment contract)
    cache.get("alpha value one")
    assert calls == ["alpha value one"]


def test_subs_clamp_concat_and_probe_echo() -> None:
    # Mixed strings + {"q","hyde"} dicts ride the SAME normalization: HyDE concat,
    # 4-entry clamp, empty-q skip. Pinned via the detector's probe echo (raw query
    # first, then the kept probe texts).
    cache, _, _ = _two_unit_world(gap_detector=True)
    r = cache.get("alpha value one", subs=[
        "beta kappa",                                     # bare string
        {"q": "junk", "hyde": "beta kappa two"},          # concat embeds "<q> <hyde>"
        {"q": "", "hyde": "dropped"},                     # empty q skipped (slot spent)
        {"q": "kappa one"},
        {"q": "beyond the clamp"},                        # 5th entry: dropped
    ])
    assert r.probes == ["alpha value one", "beta kappa",
                        "junk beta kappa two", "kappa one"]


def test_subs_malformed_fails_open_to_raw_query() -> None:
    # The callable's fail-open contract, mirrored: a malformed subs entry degrades to
    # the raw-query read (warning), never crashes and never half-applies.
    cache_off, _, _ = _two_unit_world()
    cache, _, _ = _two_unit_world()
    base = _fingerprint(cache_off.get("alpha value one"))
    assert _fingerprint(cache.get("alpha value one", subs=[42])) == base  # type: ignore[list-item]


def test_upstream_params_fail_loud_off_pool_and_on_bad_types() -> None:
    ret = _WordRetriever()
    ret.set("src:a", "alpha value gamma.")
    unit_cache = SemanticCache(ret, _ClaimSynth(["alpha value"]),     # type: ignore[arg-type]
                               embedder=FunctionEmbedder(_embed),
                               hit_threshold=0.5, coverage_floor=0.0)
    # structurally inert on the unit path -> fail LOUD (query_keys/decompose precedent)
    with pytest.raises(ValueError, match="require read_path='pool'"):
        unit_cache.get("alpha value", subs=["beta"])
    with pytest.raises(ValueError, match="require read_path='pool'"):
        unit_cache.get("alpha value", constraints={"sources": ["TechCrunch"]})
    # explicit caller params get TYPE errors, not silent degradation
    pool_cache, _, _ = _two_unit_world()
    with pytest.raises(TypeError, match="subs must be a list"):
        pool_cache.get("alpha value", subs="beta kappa")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="constraints must be a dict"):
        pool_cache.get("alpha value", constraints=["TechCrunch"])  # type: ignore[arg-type]


# ------------------------------------------------------------------- constraints= feeder

_A_SENT = "The alpha value metric reached more units in June."
_B_SENT = "Shipments of beta hardware hit many units worldwide this quarter."
_A_META = {"title": "Alpha metric quarterly report", "source": "TechCrunch",
           "date": "2023-11-02"}
_B_META = {"title": "Beta hardware shipments dossier", "source": "The Verge",
           "date": "November 5, 2023"}


def _meta_world(**kw: object) -> tuple[SemanticCache, str, str]:
    ret = _WordRetriever()
    ret.set("src:a", _A_SENT, meta=_A_META)
    ret.set("src:b", _B_SENT, meta=_B_META)
    synth = _ClaimSynth(["alpha value"])
    kw.setdefault("serve_budget", 200)
    cache = _pool(ret, synth, **kw)
    ra = cache.get("alpha value")
    synth.claims = ["beta kappa"]
    rb = cache.get("beta hardware")
    assert ra.unit_id != rb.unit_id
    return cache, ra.unit_id, rb.unit_id


def test_constraints_match_banks_right_units_spans() -> None:
    # A planted source match (case-insensitive substring vs source_meta) banks the
    # MATCHED unit's best evidence spans into the read ledger — feeder only.
    ev: list[dict] = []
    cache, a, _b = _meta_world(on_event=ev.append)
    r = cache.get("alpha value one", constraints={"sources": ["techcrunch"]})
    bank = cache._read_bank[r.read_id]
    assert [(e["origin"], e["kind"], e["unit_id"], e["span"], e["source"])
            for e in bank] == [("constraints", "metadata", a, _A_SENT, "src:a")]
    assert len(bank) <= _SPANS_PER_READ                   # per-unit candidate cap
    matched = [e for e in ev if e["event"] == "constraints_matched"]
    assert matched and matched[-1]["units"] == [a] and matched[-1]["banked"] == 1


def test_constraints_date_and_entity_forms() -> None:
    # dates: ISO constraint matches a 'Month D, YYYY' meta date THROUGH the one
    # canonicalizer (and the reverse); entities: substring vs title.
    cache, a, b = _meta_world()
    r1 = cache.get("alpha value one", constraints={"dates": ["2023-11-05"]})
    assert [e["unit_id"] for e in cache._read_bank[r1.read_id]] == [b]
    r2 = cache.get("alpha value one", constraints={"dates": ["November 2, 2023"]})
    assert [e["unit_id"] for e in cache._read_bank[r2.read_id]] == [a]
    r3 = cache.get("alpha value one", constraints={"entities": ["beta hardware"]})
    assert [e["unit_id"] for e in cache._read_bank[r3.read_id]] == [b]


def test_constraints_no_match_is_clean_noop() -> None:
    cache, _a, _b = _meta_world()
    base = _fingerprint(cache.get("alpha value one"))
    r = cache.get("alpha value one", constraints={"sources": ["Reuters"],
                                                  "dates": ["1999-01-01"]})
    assert _fingerprint(r) == base
    assert r.read_id not in cache._read_bank              # nothing banked at all


def test_constraints_zero_serving_delta_ever() -> None:
    # THE covenant pin: even a MATCHING constraint changes nothing about serving —
    # payload bytes, served claims, coverage, gate outcome all byte-identical to the
    # unconstrained read. Constraints feed the ledger; the ranker serves.
    cache, _a, _b = _meta_world()
    for q in ("alpha value one", "beta hardware", "alpha gamma"):
        base = _fingerprint(cache.get(q))
        r = cache.get(q, constraints={"sources": ["TechCrunch", "the verge"],
                                      "dates": ["2023-11-02", "November 5, 2023"],
                                      "entities": ["alpha metric", "beta hardware"]})
        assert _fingerprint(r) == base
        assert cache._read_bank[r.read_id]                # ...while the feeder DID fire


def test_constraints_unknown_key_warns(caplog: pytest.LogCaptureFixture) -> None:
    cache, _a, _b = _meta_world()
    with caplog.at_level(logging.WARNING, logger="coalent.semantic.cache"):
        cache.get("alpha value one", constraints={"date": ["2023-11-02"]})  # typo'd key
    assert any("unknown keys" in rec.message for rec in caplog.records)


# --------------------------- AND-constraints policy (v0.7b, composition verdict #2)
# AND across provided keys / OR within a key's values, applied when >= 2 DERIVABLE
# keys; single key = today's OR byte-identical; never-empty OR-fallback guard;
# range-phrase dates underivable -> key skipped, never guessed.

_A2_META = {"title": "Alpha metric quarterly report", "source": "TechCrunch",
            "date": "2023-10-06"}
_B2_META = {"title": "Beta hardware shipments dossier", "source": "Cnbc",
            "date": "2023-10-06"}          # SAME date, wrong source — the q476 caption shape


def _and_world(**kw: object) -> tuple[SemanticCache, str, str]:
    ret = _WordRetriever()
    ret.set("src:a", _A_SENT, meta=_A2_META)
    ret.set("src:b", _B_SENT, meta=_B2_META)
    synth = _ClaimSynth(["alpha value"])
    kw.setdefault("serve_budget", 200)
    cache = _pool(ret, synth, **kw)
    ra = cache.get("alpha value")
    synth.claims = ["beta kappa"]
    rb = cache.get("beta hardware")
    assert ra.unit_id != rb.unit_id
    return cache, ra.unit_id, rb.unit_id


def test_and_excludes_same_date_wrong_source_unit() -> None:
    # THE q476/NEWS-02 shape: under OR, a same-date wrong-source unit (the courtroom-
    # sketch caption unit's geometry) feeds the repair queue; under >=2-keys AND it is
    # excluded while the named-source gold unit stays.
    cache, a, b = _and_world()
    r = cache.get("alpha value one",
                  constraints={"dates": ["2023-10-06"], "sources": ["TechCrunch"]})
    assert {e["unit_id"] for e in cache._read_bank[r.read_id]} == {a}
    # sources absent -> single derivable key -> OR (today's semantics): the same-date
    # wrong-source unit is included again.
    r2 = cache.get("alpha value one", constraints={"dates": ["2023-10-06"]})
    assert {e["unit_id"] for e in cache._read_bank[r2.read_id]} == {a, b}


def test_and_zero_match_falls_back_to_or_with_event() -> None:
    # THE q272 shape: the dates value parses to a day NO unit carries, so AND = empty
    # -> the never-empty guard recomputes under OR (recovers the source-named unit)
    # and logs constraints_and_fallback.
    ev: list[dict] = []
    cache, a, _b = _and_world(on_event=ev.append)
    r = cache.get("alpha value one",
                  constraints={"dates": ["2023-11-05"], "sources": ["techcrunch"]})
    assert {e["unit_id"] for e in cache._read_bank[r.read_id]} == {a}
    fb = [e for e in ev if e["event"] == "constraints_and_fallback"]
    assert fb and fb[-1]["keys"] == 2 and fb[-1]["or_matched"] == 1


def test_and_range_phrase_date_key_skipped_never_guessed() -> None:
    # An unparseable range-phrase date makes the dates key UNDERIVABLE: the key is
    # skipped (single effective key -> OR), it is NOT a failed conjunct and NOT a
    # fallback — no constraints_and_fallback event fires.
    ev: list[dict] = []
    cache, a, _b = _and_world(on_event=ev.append)
    r = cache.get("alpha value one", constraints={
        "dates": ["between October 6 and October 7, 2023"],
        "sources": ["TechCrunch"]})
    assert {e["unit_id"] for e in cache._read_bank[r.read_id]} == {a}
    assert not [e for e in ev if e["event"] == "constraints_and_fallback"]


def test_and_single_key_byte_identical_to_today() -> None:
    # Single-key calls keep today's OR behavior exactly: same banked rows (shape and
    # content as the shipped feeder) AND serving byte-identical (feeder-only covenant).
    cache, a, b = _and_world()
    base = _fingerprint(cache.get("alpha value one"))
    r_src = cache.get("alpha value one", constraints={"sources": ["cnbc"]})
    assert _fingerprint(r_src) == base
    assert [(e["origin"], e["kind"], e["unit_id"], e["span"], e["source"])
            for e in cache._read_bank[r_src.read_id]] \
        == [("constraints", "metadata", b, _B_SENT, "src:b")]
    r_dt = cache.get("alpha value one", constraints={"dates": ["October 6, 2023"]})
    assert _fingerprint(r_dt) == base
    assert {e["unit_id"] for e in cache._read_bank[r_dt.read_id]} == {a, b}


def test_and_matching_keys_zero_serving_delta() -> None:
    # The feeder-only covenant holds under the AND policy too: a >=2-keys read that
    # banks candidates serves byte-identically to the unconstrained read.
    cache, a, _b = _and_world()
    base = _fingerprint(cache.get("alpha value one"))
    r = cache.get("alpha value one",
                  constraints={"dates": ["2023-10-06"], "sources": ["TechCrunch"]})
    assert _fingerprint(r) == base
    assert {e["unit_id"] for e in cache._read_bank[r.read_id]} == {a}


# ------------------------------------------------------------- the ONE date canonicalizer

def test_canonical_date_forms() -> None:
    iso = "2023-11-02"
    # ISO in, ISO out (zero-padded), time tails tolerated
    assert _canonical_date("2023-11-02") == iso
    assert _canonical_date("2023-11-2") == iso
    assert _canonical_date("2023-11-02T10:30:00Z") == iso
    # 'Month D, YYYY' — full month, >=3-letter prefix, period, ordinal, comma optional
    assert _canonical_date("November 2, 2023") == iso
    assert _canonical_date("Nov 2, 2023") == iso
    assert _canonical_date("Nov. 2 2023") == iso
    assert _canonical_date("November 2nd, 2023") == iso
    # never guesses: unparseable/ambiguous forms canonicalize to "" (can't match)
    assert _canonical_date("Janet 5, 2020") == ""         # not a month prefix
    assert _canonical_date("13/05/2023") == ""
    assert _canonical_date("May 2023") == ""              # no day
    assert _canonical_date("2023-13-40") == ""            # invalid month/day
    assert _canonical_date("") == ""
