"""v0.6 — tier-2 residual spans + the repair loop (opt-in ``residual_spans``).

The measured basis: harmful extraction loss 50-84% on news, re-injection
causal flips 56%, sibling masking makes the loss SILENT — and the chunked-dedup verdict says
news pools must stay LEAN, so the net must add ZERO rows to the main claim pool. The design:

* BUILD-TIME SENTENCE-COVERAGE AUDIT — fact-bearing evidence sentences (a number or a
  multi-word capitalized name; >= 6 words; boilerplate excluded) whose max cosine against the
  unit's claims is below ``span_tau`` are retained on the unit as tier-2 spans, with
  provenance (artifact + evidence-chunk index) captured at build, never re-derived.
* SIDE-CHANNEL SERVING — spans are NEVER pool rows; under the ranking-anomaly rule (span
  similarity > best fresh-claim similarity + ``span_margin``) up to 2 spans per read serve as
  labeled "[source excerpt]" lines under their owner's attribution, inside serve_budget.
* LOSSY SIGNALS + REPAIR — span serves and raw-fallback escalations count toward
  ``lossy_threshold``; a lossy-marked unit repairs on its next rebuild touch with THE
  APPEND-ONLY INVARIANT: source hash unchanged -> union(old, new) claims deduped at 0.95
  (old claims never lost); hash changed -> full replace (existing stale behavior).

All of it is default-OFF: ``residual_spans=False`` is pinned byte-identical below.
"""
from __future__ import annotations

import json
import re

import pytest

from coalent import FunctionEmbedder
from coalent.domain.models import ProvenanceManifest, SourceSpan
from coalent.semantic import (
    Chunk,
    Cognition,
    QueryKey,
    ResidualSpan,
    SemanticCache,
    Synthesis,
)
from coalent.semantic import cache as cache_module
from coalent.semantic.serde import (
    cognition_from_dict,
    cognition_from_json,
    cognition_to_dict,
    cognition_to_json,
)

_AXES = ("alpha", "beta", "gamma", "kappa", "one", "two", "three", "value", "junk")
_STRIP = re.compile(r"[^\w\s]")


def _embed(text: str) -> list[float]:
    words = set(_STRIP.sub(" ", text.lower()).split())
    v = [1.0 if a in words else 0.0 for a in _AXES]
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v] if n else v


class _ClaimSynth:
    """Returns a FIXED claim list (mutable between calls) — so the sentence-coverage audit
    and the append-only merge are driven deterministically, independent of chunk text."""

    def __init__(self, claims: list[str]) -> None:
        self.claims = list(claims)
        self.calls = 0

    def synthesize(self, query: str, chunks: list[Chunk]) -> Synthesis:
        self.calls += 1
        return Synthesis(understanding={"claims": list(self.claims)},
                         used=list(range(len(chunks))))


class _WordRetriever:
    """Word-overlap retriever with editable sources — drives contained probes (same text)
    and silent-drift changed probes (edited text, no source_changed event)."""

    def __init__(self) -> None:
        self.docs: dict[str, str] = {}
        self.calls = 0

    def set(self, artifact_id: str, text: str) -> None:
        self.docs[artifact_id] = text

    def retrieve(self, query: str, *, namespace: str | None = None) -> list[Chunk]:
        self.calls += 1
        qs = set(_STRIP.sub(" ", query.lower()).split())
        return [Chunk(artifact_id=aid, text=text) for aid, text in self.docs.items()
                if qs & set(_STRIP.sub(" ", text.lower()).split())]


# One news-shaped doc: a covered fact sentence, two uncovered fact sentences (number-bearing
# and name-bearing), number-bearing boilerplate, a short fragment, and a non-fact sentence.
_DOC = (
    "The alpha value metric reached 42 units in June. "
    "Shipments of beta hardware hit 900 units worldwide this quarter. "
    "Marcus Chen briefed the gamma committee about the merger. "
    "Subscribe to our beta newsletter for 10 free reports. "
    "beta gained 5. "
    "everyone seemed rather pleased with beta overall."
)
_BETA_SPAN = "Shipments of beta hardware hit 900 units worldwide this quarter."
_NAME_SPAN = "Marcus Chen briefed the gamma committee about the merger."


def _pool(retriever: object, synth: object, **kw: object) -> SemanticCache:
    kw.setdefault("coverage_floor", 0.0)
    kw.setdefault("residual_spans", True)
    kw.setdefault("serve_gate", 0.5)
    return SemanticCache(retriever, synth,                            # type: ignore[arg-type]
                         embedder=FunctionEmbedder(_embed),
                         read_path="pool", **kw)                      # type: ignore[arg-type]


# ------------------------------------------------------------------- build-time audit

def test_residual_spans_captured_at_build() -> None:
    ret = _WordRetriever()
    ret.set("src:news", _DOC)
    synth = _ClaimSynth(["alpha value"])
    cache = SemanticCache(ret, synth,                                 # type: ignore[arg-type]
                          embedder=FunctionEmbedder(_embed),
                          hit_threshold=0.99, coverage_floor=0.0, residual_spans=True,
                          read_path="unit")
    r = cache.get("alpha value")
    unit = cache._units[r.unit_id]
    # Exactly the two UNCOVERED fact-bearing sentences: the covered one (claim cosine 1.0),
    # the boilerplate (despite its number), the short fragment, and the non-fact sentence
    # are all excluded. Document order preserved.
    assert [s.text for s in unit.residual_spans] == [_BETA_SPAN, _NAME_SPAN]
    for span in unit.residual_spans:
        assert span.artifact_id == "src:news"          # provenance captured at build
        assert span.chunk_idx == 0                     # index into unit.evidence
        assert unit.evidence[span.chunk_idx].text == _DOC
        assert span.embedding == tuple(_embed(span.text))   # embedded once, persisted
    # Spans live BESIDE the claims, never inside them.
    assert unit.understanding["claims"] == ["alpha value"]


def _unit_cache(ret: _WordRetriever, synth: _ClaimSynth) -> SemanticCache:
    return SemanticCache(ret, synth,                                  # type: ignore[arg-type]
                         embedder=FunctionEmbedder(_embed),
                         hit_threshold=0.99, coverage_floor=0.0, residual_spans=True,
                         read_path="unit")


def test_capture_requires_absent_hard_facts() -> None:
    # The eyeball-audit fix: a sentence UNDER span_tau by cosine whose hard facts (multi-char
    # numbers, multi-word capitalized names) ALL appear in some claim text is a phrasing gap,
    # not extraction loss — it must NOT be captured. Only sentences adding an absent fact are.
    ret = _WordRetriever()
    ret.set("src:news",
            "Kappa shipments hit 900 units this quarter. "      # 900 in claim -> covered
            "Kappa shipments hit 901 units this quarter. "      # 901 absent   -> captured
            "Marcus Chen visited the kappa office in Denver. "  # name in claim -> covered
            "Sarah Jones visited the kappa office again today.")  # name absent -> captured
    synth = _ClaimSynth(["alpha value shipments totaled 900 for Marcus Chen"])
    cache = _unit_cache(ret, synth)
    r = cache.get("kappa")
    unit = cache._units[r.unit_id]
    assert [s.text for s in unit.residual_spans] == [
        "Kappa shipments hit 901 units this quarter.",
        "Sarah Jones visited the kappa office again today.",
    ]


def test_capture_rejects_segmentation_orphans() -> None:
    ret = _WordRetriever()
    ret.set("src:news",
            "Coverage begins at 11 p.m. "
            "ET\n\nPotential Super Bowl matchups excite beta fans heading into 2026. "
            "and then 500 more beta units shipped worldwide yesterday. "
            "Beta shipments reached 800 units\n\nMore beta coverage arrived today.")
    cache = _unit_cache(ret, _ClaimSynth(["alpha value"]))
    r = cache.get("beta")
    unit = cache._units[r.unit_id]
    # The stranded "ET\n\n" splitter orphan is stripped, leaving the clean sentence; the
    # lowercase-start fragment and the embedded-blank-line fragment are rejected outright.
    assert [s.text for s in unit.residual_spans] == [
        "Potential Super Bowl matchups excite beta fans heading into 2026."]
    assert all("\n" not in s.text and not s.text.startswith("ET") for s in unit.residual_spans)


def test_span_cap_keeps_most_uncovered() -> None:
    # 14 uncovered fact sentences; the partially-covered one (claim cosine 0.5, still under
    # span_tau) sits FIRST in the document — the cap must drop it (and the 13th zero-cosine
    # sentence), keeping the 12 with the LOWEST best-claim cosine, most clearly uncovered.
    partial = "The alpha backlog dropped 77 kappa points overall."
    zeros = [f"Beta shipment number {100 + i} arrived at dock number {200 + i} today."
             for i in range(13)]
    ret = _WordRetriever()
    ret.set("src:news", " ".join([partial] + zeros))
    cache = _unit_cache(ret, _ClaimSynth(["alpha value"]))
    r = cache.get("beta kappa")
    unit = cache._units[r.unit_id]
    assert len(unit.residual_spans) == 12
    assert [s.text for s in unit.residual_spans] == zeros[:12]   # ties keep document order
    texts = {s.text for s in unit.residual_spans}
    assert partial not in texts and zeros[12] not in texts


def test_spans_survive_serde_roundtrip() -> None:
    span = ResidualSpan(text=_BETA_SPAN, artifact_id="src:news", chunk_idx=0,
                        embedding=(0.0, 1.0, 0.5))
    unit = Cognition(
        id="cog:x", namespace="", query="alpha", query_embedding=(1.0, 0.0),
        understanding={"claims": ["alpha value"]}, evidence=(),
        provenance=ProvenanceManifest("s", "p"),
        residual_spans=(span,), span_hits=3, lossy=True,
    )
    restored = cognition_from_json(cognition_to_json(unit))
    # Text + provenance round-trip exactly; the embedding is NOT persisted (it is
    # regenerable — re-embedded in one batch on first side-channel use after load).
    assert restored.residual_spans == (ResidualSpan(
        text=_BETA_SPAN, artifact_id="src:news", chunk_idx=0, embedding=()),)
    assert restored.span_hits == 3 and restored.lossy is True

    # An OLD stored dict (pre-v0.6, no span fields) loads fine with clean defaults.
    data = json.loads(cognition_to_json(unit))
    for key in ("residual_spans", "span_hits", "lossy"):
        data.pop(key)
    old = cognition_from_dict(data)
    assert old.residual_spans == () and old.span_hits == 0 and old.lossy is False

    # And a unit that never opted in writes NO new keys — v0.5 JSON stays byte-identical.
    plain = cognition_from_dict(data)
    assert not {"residual_spans", "span_hits", "lossy"} & set(cognition_to_dict(plain))


def test_span_serde_is_lightweight() -> None:
    ret = _WordRetriever()
    ret.set("src:news", _DOC)
    cache = SemanticCache(ret, _ClaimSynth(["alpha value"]),          # type: ignore[arg-type]
                          embedder=FunctionEmbedder(_embed),
                          hit_threshold=0.99, coverage_floor=0.0, residual_spans=True,
                          read_path="unit")
    r = cache.get("alpha value")
    unit = cache._units[r.unit_id]
    assert unit.residual_spans
    assert all(s.embedding for s in unit.residual_spans)   # freshly built: in-memory embs
    for entry in cognition_to_dict(unit)["residual_spans"]:
        assert set(entry) == {"text", "artifact_id", "chunk_idx"}   # no ~30KB embedding key
    # A legacy dict that DOES carry an embedding still loads it (read-compat).
    legacy_span = {"text": _BETA_SPAN, "artifact_id": "src:news", "chunk_idx": 0,
                   "embedding": [0.0, 1.0]}
    data = cognition_to_dict(unit)
    data["residual_spans"] = [legacy_span]
    assert cognition_from_dict(data).residual_spans[0].embedding == (0.0, 1.0)


def test_spans_rehydrate_after_reload() -> None:
    from coalent.semantic import SQLiteCognitionStore

    store = SQLiteCognitionStore(":memory:")           # JSON serde on every put/all
    ret = _WordRetriever()
    ret.set("src:news", _DOC)
    warm = _pool(ret, _ClaimSynth(["alpha value"]), store=store)
    warm.get("alpha value")                            # build + persist (spans sans embs)

    ret2 = _WordRetriever()
    ret2.set("src:news", _DOC)
    cold = _pool(ret2, _ClaimSynth(["alpha value"]), store=store)   # a fresh process
    unit = next(iter(cold._units.values()))
    assert unit.residual_spans
    assert all(s.embedding == () for s in unit.residual_spans)      # loaded lightweight
    r = cold.get("beta two")                           # first side-channel use hydrates
    assert "- [source excerpt] " + _BETA_SPAN in r.context["pool"]
    assert all(s.embedding for s in unit.residual_spans)            # recomputed, in-memory


# --------------------------------------------------------------- zero pool crowding

def test_spans_never_enter_claim_pool() -> None:
    ret = _WordRetriever()
    ret.set("src:news", _DOC)
    cache = _pool(ret, _ClaimSynth(["alpha value"]))
    cache.get("alpha value")                           # build: 1 claim + 2 tier-2 spans
    unit = next(iter(cache._units.values()))
    assert len(unit.residual_spans) == 2               # the net exists...
    idx = cache._claim_indexes[""]
    assert len(idx) == 1                               # ...but added ZERO pool rows
    hits = idx.search(_embed("beta two"), 10)
    assert [ref.text for _s, ref, _f in hits] == ["alpha value"]
    assert cache.stats()["pool_claims_total"] == 1     # stats see claims only


# ------------------------------------------------------------ side-channel serving

def test_span_anomaly_serves_labeled_span() -> None:
    events: list[dict] = []
    ret = _WordRetriever()
    ret.set("src:news", _DOC)
    cache = _pool(ret, _ClaimSynth(["alpha value"]), on_event=events.append,
                  pool_header=lambda u: "## NEWS")
    cache.get("alpha value")                           # build
    unit = next(iter(cache._units.values()))
    r = cache.get("beta two")                          # claim cos 0; beta span cos ~0.707
    assert r.cache_hit is True                         # a serve read — no synthesis paid
    # Served, labeled, and attributed UNDER the owner's pool_header group.
    assert "## NEWS\n- alpha value\n- [source excerpt] " + _BETA_SPAN in r.context["pool"]
    assert r.context["pool"].count("[source excerpt]") == 1   # gamma span is no anomaly
    # The span never leaks into the claim payload.
    assert _BETA_SPAN not in r.understanding["claims"]
    assert all(p.claim != _BETA_SPAN for p in r.pool)
    served = [e for e in events if e["event"] == "residual_served"]
    assert len(served) == 1
    assert served[0]["unit_id"] == unit.id
    assert served[0]["span_sim"] == pytest.approx(2.0 ** -0.5, abs=1e-3)
    assert served[0]["claim_sim"] == pytest.approx(0.0, abs=1e-6)
    assert unit.span_hits == 1 and unit.lossy is False
    # Budget-respected: the span costs inside the SAME serve_budget accounting.
    pool_served = [e for e in events if e["event"] == "pool_served"][-1]
    assert pool_served["est_tokens"] <= pool_served["budget"]

    # A span that does not FIT the budget is not served (the net never overruns).
    events2: list[dict] = []
    ret2 = _WordRetriever()
    ret2.set("src:news", _DOC)
    tight = _pool(ret2, _ClaimSynth(["alpha value"]), on_event=events2.append,
                  pool_header=lambda u: "## NEWS", serve_budget=6)
    tight.get("alpha value")
    r2 = tight.get("beta two")
    assert "[source excerpt]" not in r2.context["pool"]
    assert not [e for e in events2 if e["event"] == "residual_served"]

    # span_margin raises the anomaly bar: no serve when the span cannot clear it.
    ret3 = _WordRetriever()
    ret3.set("src:news", _DOC)
    wide = _pool(ret3, _ClaimSynth(["alpha value"]), span_margin=0.8)
    wide.get("alpha value")
    r3 = wide.get("beta two")
    assert "[source excerpt]" not in r3.context["pool"]


def test_span_serve_capped_at_two() -> None:
    events: list[dict] = []
    ret = _WordRetriever()
    ret.set("src:news",
            "Shipments of beta hardware hit 900 units worldwide. "
            "Analysts said beta output reached 500 units in March. "
            "Suppliers of beta panels shipped 70 crates in April.")
    cache = _pool(ret, _ClaimSynth(["junk"]), on_event=events.append)
    cache.get("junk beta")                             # build: 1 claim, 3 uncovered spans
    r = cache.get("beta two")                          # all 3 spans outrank every claim
    assert r.context["pool"].count("[source excerpt]") == 2   # hard cap per read
    assert len([e for e in events if e["event"] == "residual_served"]) == 2


def test_span_side_channel_pure_python(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache_module, "_np", None)     # covenant: pure-python twin serves too
    ret = _WordRetriever()
    ret.set("src:news", _DOC)
    cache = _pool(ret, _ClaimSynth(["alpha value"]))
    cache.get("alpha value")
    r = cache.get("beta two")
    assert "- [source excerpt] " + _BETA_SPAN in r.context["pool"]


# ------------------------------------------------------------------- lossy signals

def test_span_hits_mark_unit_lossy() -> None:
    events: list[dict] = []
    ret = _WordRetriever()
    ret.set("src:news", _DOC)
    cache = _pool(ret, _ClaimSynth(["alpha value"]), on_event=events.append)
    cache.get("alpha value")                           # build
    unit = next(iter(cache._units.values()))
    cache.get("beta two")                              # span serve #1
    assert unit.span_hits == 1 and unit.lossy is False
    assert not [e for e in events if e["event"] == "unit_marked_lossy"]
    cache.get("beta one")                              # span serve #2 -> lossy_threshold=2
    assert unit.span_hits == 2 and unit.lossy is True
    marked = [e for e in events if e["event"] == "unit_marked_lossy"]
    assert len(marked) == 1                            # marked ONCE, never re-emitted
    assert marked[0] == {"event": "unit_marked_lossy", "ts": marked[0]["ts"],
                         "unit_id": unit.id, "reason": "span_hits"}


def test_raw_fallback_marks_unit_lossy() -> None:
    events: list[dict] = []
    ret = _WordRetriever()
    ret.set("src:a", "The alpha value figure was 42 in June.")   # fully covered: no spans
    cache = _pool(ret, _ClaimSynth(["alpha value"]), on_event=events.append,
                  coverage_floor=0.9, lossy_threshold=1)
    cache.get("alpha value")                           # build; cov 1.0 -> no escalation
    unit = next(iter(cache._units.values()))
    assert unit.residual_spans == ()
    r = cache.get("alpha two")                         # serve (cov 0.5 >= gate) but < floor
    assert r.cache_hit is True and r.escalated is True
    # The served raw chunk's artifact is owned by this FRESH unit -> escalation feedback.
    assert unit.span_hits == 1 and unit.lossy is True
    marked = [e for e in events if e["event"] == "unit_marked_lossy"]
    assert len(marked) == 1 and marked[0]["reason"] == "raw_fallback"
    assert marked[0]["unit_id"] == unit.id


# --------------------------------------------------------------------- repair loop

def test_append_only_repair_unchanged_snapshot() -> None:
    events: list[dict] = []
    ret = _WordRetriever()
    ret.set("src:a", "The alpha report cited 12 kappa readings overall.")
    synth = _ClaimSynth(["alpha one", "alpha two"])
    cache = _pool(ret, synth, serve_gate=0.9, on_event=events.append, span_tau=0.01)
    r1 = cache.get("alpha value")                      # build with claims 1, 2
    unit = cache._units[r1.unit_id]
    unit.lossy = True                                  # lossy-marked (signals pinned above)
    # The repair extraction returns an exact dup, a 0.95-near dup, and one NEW claim.
    synth.claims = ["alpha one", "one alpha", "alpha three"]
    r2 = cache.get("alpha value")                      # next touch: P1 reuse -> repair
    assert r2.cache_hit is False                       # the repair paid one synthesis
    # THE APPEND-ONLY INVARIANT: source hash unchanged -> old claims 1, 2 survive (order
    # kept), claim 3 is added, dups collapse within-unit at 0.95. Coverage is monotone.
    assert unit.understanding["claims"] == ["alpha one", "alpha two", "alpha three"]
    repaired = [e for e in events if e["event"] == "unit_repaired"]
    assert len(repaired) == 1
    assert repaired[0]["unit_id"] == unit.id
    assert repaired[0]["added_claims"] == 1 and repaired[0]["kept_claims"] == 2
    assert unit.lossy is False and unit.span_hits == 0   # repaired -> signals cleared
    # No duplicates in the pool after repair (add() replaced the unit's rows atomically).
    idx = cache._claim_indexes[""]
    assert len(idx) == 3
    texts = [ref.text for _s, ref, _f in idx.search(_embed("alpha"), 10)]
    assert sorted(texts) == ["alpha one", "alpha three", "alpha two"]
    r3 = cache.get("alpha one")                        # and the merged unit SERVES all three
    for claim in ("alpha one", "alpha two", "alpha three"):
        assert "- " + claim in r3.context["pool"]


def test_changed_snapshot_replaces() -> None:
    events: list[dict] = []
    ret = _WordRetriever()
    ret.set("src:a", "The alpha report cited 12 kappa readings overall.")
    synth = _ClaimSynth(["alpha one", "alpha two"])
    cache = _pool(ret, synth, serve_gate=0.9, on_event=events.append, span_tau=0.01)
    r1 = cache.get("alpha value")
    unit = cache._units[r1.unit_id]
    unit.lossy = True
    # SILENT DRIFT: the source text changes with no source_changed event — the provenance
    # hash check must catch it and fall back to the full replace (existing stale behavior).
    ret.set("src:a", "The alpha report cited 99 kappa readings overall.")
    synth.claims = ["alpha three"]
    r2 = cache.get("alpha value")                      # touch -> repair -> hash CHANGED
    assert r2.cache_hit is False
    assert unit.understanding["claims"] == ["alpha three"]   # replaced, not merged
    repaired = [e for e in events if e["event"] == "unit_repaired"]
    assert len(repaired) == 1
    assert repaired[0]["added_claims"] == 1 and repaired[0]["kept_claims"] == 0
    assert unit.lossy is False and unit.span_hits == 0
    idx = cache._claim_indexes[""]
    assert len(idx) == 1
    assert [ref.text for _s, ref, _f in idx.search(_embed("alpha"), 10)] == ["alpha three"]


def test_contained_probe_repairs_lossy_unit() -> None:
    # The P5 touch: a CONTAINED probe normally skips (provenance proves coverage) — but a
    # lossy containing owner must NOT be skipped; the probe is its repair touch.
    events: list[dict] = []
    ret = _WordRetriever()
    ret.set("src:a", "The alpha report cited 12 kappa readings overall.")
    synth = _ClaimSynth(["alpha one", "alpha two"])
    cache = _pool(ret, synth, serve_gate=0.9, on_event=events.append, span_tau=0.01)
    r1 = cache.get("alpha value")
    unit = cache._units[r1.unit_id]
    unit.lossy = True
    synth.claims = ["alpha one", "alpha three"]
    r2 = cache.get("alpha kappa")                      # seed cos 0.5: NOT reuse; below gate
    assert r2.cache_hit is False                       # probe -> contained-but-lossy -> repair
    assert unit.understanding["claims"] == ["alpha one", "alpha two", "alpha three"]
    repaired = [e for e in events if e["event"] == "unit_repaired"]
    assert len(repaired) == 1 and repaired[0]["kept_claims"] == 2
    assert not [e for e in events if e["event"] == "admission_reuse"]
    assert len(cache._units) == 1                      # rebuilt in place, no duplicate unit


# ------------------------------------------- behavioral fallback channel (report_refusal)

def test_repair_chain_end_to_end() -> None:
    # The LIVE chain, no mocks: extraction drops a fact -> sibling masking keeps the serve
    # confident (anomaly silent) -> the ANSWERER's refusals (the behavioral channel) mark
    # the unit lossy -> the next probe touch repairs append-only -> the missed fact is a
    # claim, the net retires. The measured story, end to end.
    events: list[dict] = []
    ret = _WordRetriever()
    ret.set("src:a", "Alpha beta value coverage improved. "
                     "Beta shipments hit 900 units in kappa region.")
    missed = "Beta shipments hit 900 units in kappa region."
    synth = _ClaimSynth(["alpha beta value"])           # extraction PROVABLY drops the 900 fact
    cache = _pool(ret, synth, on_event=events.append, pool_header=lambda u: "## SRC")

    r0 = cache.get("alpha beta value")                  # build
    unit = cache._units[r0.unit_id]
    assert "900" in unit.evidence[0].text               # the fact IS in the evidence...
    assert all("900" not in c for c in unit.understanding["claims"])   # ...not in the claims
    assert [s.text for s in unit.residual_spans] == [missed]           # the audit caught it
    e_build = cache._rows_epoch
    assert e_build >= 1

    # Read 1: sibling masking — claim cosine 0.82 serves confidently, the span (0.5) is NO
    # ranking anomaly. The answerer refuses; the refusal is the trigger rank couldn't be.
    r1 = cache.get("beta value")
    assert r1.cache_hit is True
    assert "[source excerpt]" not in r1.context["pool"]
    p1 = cache.report_refusal(r1.read_id)
    assert p1 == "## SRC\n- [source excerpt] " + missed   # the attributed retry payload
    assert unit.span_hits == 1 and unit.lossy is False

    # Read 2: same shape -> second behavioral residual-hit crosses lossy_threshold.
    r2 = cache.get("beta value one")
    assert "[source excerpt]" not in r2.context["pool"]
    p2 = cache.report_refusal(r2.read_id)
    assert p2 == p1
    assert unit.span_hits == 2 and unit.lossy is True
    marked = [e for e in events if e["event"] == "unit_marked_lossy"]
    assert len(marked) == 1 and marked[0]["reason"] == "span_hits"
    fallbacks = [e for e in events if e["event"] == "residual_fallback"]
    assert [f["read_id"] for f in fallbacks] == [r1.read_id, r2.read_id]
    assert all(f["unit_id"] == unit.id for f in fallbacks)
    assert cache._rows_epoch == e_build                 # refusals are $0: no rows touched

    # Read 3: the next touching read (probe path) repairs APPEND-ONLY on the unchanged hash.
    synth.claims = [missed]                             # the repair extraction finds the fact
    r3 = cache.get("beta kappa")                        # below gate -> probe -> contained+lossy
    assert r3.cache_hit is False
    assert unit.understanding["claims"] == ["alpha beta value", missed]   # old kept + fact added
    repaired = [e for e in events if e["event"] == "unit_repaired"]
    assert len(repaired) == 1
    assert repaired[0] == {"event": "unit_repaired", "ts": repaired[0]["ts"],
                           "unit_id": unit.id, "added_claims": 1, "kept_claims": 1}
    assert unit.lossy is False and unit.span_hits == 0
    assert cache._rows_epoch > e_build                  # rows_epoch stays monotone
    idx = cache._claim_indexes[""]
    assert len(idx) == 2                                # ZERO duplicate rows for the unit
    texts = [ref.text for _s, ref, _f in idx.search(_embed("beta"), 10)]
    assert sorted(texts) == sorted(["alpha beta value", missed])
    assert unit.residual_spans == ()                    # the fact graduated: the net retires


def test_report_refusal_unknown_id_safe() -> None:
    events: list[dict] = []
    ret = _WordRetriever()
    ret.set("src:news", _DOC)
    cache = _pool(ret, _ClaimSynth(["alpha value"]), on_event=events.append)
    assert cache.report_refusal("bogus") is None        # unknown id: None, never raises
    r0 = cache.get("alpha value")
    for _ in range(64):                                 # roll the ring: r0 ages out
        cache.get("alpha value")
    assert cache.report_refusal(r0.read_id) is None     # expired id: None, never raises
    assert not [e for e in events if e["event"] == "residual_fallback"]

    off = SemanticCache(ret, _ClaimSynth(["alpha value"]),            # type: ignore[arg-type]
                        embedder=FunctionEmbedder(_embed), read_path="pool",
                        serve_gate=0.5, coverage_floor=0.0)           # residual_spans OFF
    r_off = off.get("alpha value")
    assert r_off.read_id                                # reads still carry an id...
    assert off.report_refusal(r_off.read_id) is None    # ...but the channel is inert


def test_fallback_respects_span_cap_and_attribution() -> None:
    doc = ("Shipments of beta hardware hit 900 units worldwide. "
           "Analysts said beta output reached 500 units in March. "
           "Suppliers of beta panels shipped 70 crates in April.")

    # Scenario A — anomaly silenced (huge span_margin): the fallback alone serves, capped.
    events: list[dict] = []
    ret = _WordRetriever()
    ret.set("src:news", doc)
    cache = _pool(ret, _ClaimSynth(["junk"]), on_event=events.append,
                  pool_header=lambda u: "## SRC", span_margin=5.0)
    cache.get("junk beta")                              # build: 3 qualifying spans
    unit = next(iter(cache._units.values()))
    r = cache.get("beta two")
    assert "[source excerpt]" not in r.context["pool"]  # the anomaly channel stayed silent
    payload = cache.report_refusal(r.read_id)
    assert payload == (
        "## SRC\n"
        "- [source excerpt] Shipments of beta hardware hit 900 units worldwide.\n"
        "- [source excerpt] Analysts said beta output reached 500 units in March.")
    assert payload.count("[source excerpt]") == 2       # the 2-span cap holds
    fallbacks = [e for e in events if e["event"] == "residual_fallback"]
    assert len(fallbacks) == 2
    assert all(f["read_id"] == r.read_id and f["unit_id"] == unit.id for f in fallbacks)
    assert unit.span_hits == 2 and unit.lossy is True   # fed exactly like the anomaly path

    # Scenario B — spans the refused payload ALREADY contained are skipped: the anomaly
    # channel served the top two, so the fallback returns only the remaining third.
    ret2 = _WordRetriever()
    ret2.set("src:news", doc)
    both = _pool(ret2, _ClaimSynth(["junk"]), pool_header=lambda u: "## SRC")
    both.get("junk beta")
    r2 = both.get("beta two")
    assert r2.context["pool"].count("[source excerpt]") == 2
    p2 = both.report_refusal(r2.read_id)
    assert p2 == ("## SRC\n"
                  "- [source excerpt] Suppliers of beta panels shipped 70 crates in April.")
    assert "Suppliers of beta panels" not in r2.context["pool"]   # genuinely NEW material


# --------------------------------------------------- query keys (behavioral alternate keys)

def _vec(**weights: float) -> list[float]:
    v = [0.0] * len(_AXES)
    for axis, w in weights.items():
        v[_AXES.index(axis)] = w
    return v


# Exact-similarity probes for the key_floor rule (0.85): unit-norm vectors whose cosine to
# the kappa-axis key embedding is exactly 0.84 / 0.86 (remainder on unrelated axes).
_SPECIAL: dict[str, list[float]] = {
    "q084": _vec(kappa=0.84, junk=(1 - 0.84 ** 2) ** 0.5),
    "q086": _vec(kappa=0.86, junk=(1 - 0.86 ** 2) ** 0.5),
    "qmix": _vec(kappa=0.86,
                 alpha=(1 - 0.86 ** 2) ** 0.5 / 2 ** 0.5,
                 value=(1 - 0.86 ** 2) ** 0.5 / 2 ** 0.5),
}


def _embed_special(text: str) -> list[float]:
    if text in _SPECIAL:
        return list(_SPECIAL[text])
    return _embed(text)


def _mk_unit(uid: str, qtext: str, claims: list[str]) -> Cognition:
    emb = tuple(_embed(qtext))
    return Cognition(
        id=uid, namespace="", query=qtext, query_embedding=emb,
        understanding={"claims": list(claims)}, evidence=(),
        provenance=ProvenanceManifest(
            "s", "p", source_spans=(SourceSpan.from_text(f"art:{uid}", "doc"),)),
        understanding_embedding=emb,
        claim_embeddings=tuple(tuple(_embed(c)) for c in claims),
    )


def _key_cache(**kw: object) -> SemanticCache:
    kw.setdefault("coverage_floor", 0.0)
    return SemanticCache(_WordRetriever(), _ClaimSynth(["junk"]),     # type: ignore[arg-type]
                         embedder=FunctionEmbedder(_embed_special), read_path="pool",
                         serve_gate=0.5, query_keys=True, **kw)       # type: ignore[arg-type]


def test_key_attaches_on_fallback_and_transfers_on_repair() -> None:
    events: list[dict] = []
    ret = _WordRetriever()
    ret.set("src:a", "Alpha beta value coverage improved. "
                     "Beta shipments hit 900 units in kappa region.")
    missed = "Beta shipments hit 900 units in kappa region."
    synth = _ClaimSynth(["alpha beta value"])
    cache = _pool(ret, synth, on_event=events.append, query_keys=True)
    r0 = cache.get("alpha beta value")                  # build; extraction drops the fact
    unit = cache._units[r0.unit_id]

    r1 = cache.get("beta value")
    assert cache.report_refusal(r1.read_id) is not None
    assert len(unit.query_keys) == 1                    # optimistic-attach on the fallback
    key = unit.query_keys[0]
    assert key.read_id == r1.read_id                    # PROVISIONAL (expires with the ring)
    assert key.claim_idx == -1                          # span-level: the fact is no claim yet
    assert key.span_text == missed
    assert key.embedding == tuple(_embed("beta value"))   # the REFUSED query is the key
    attached = [e for e in events if e["event"] == "key_attached"]
    assert attached == [{"event": "key_attached", "ts": attached[0]["ts"],
                         "unit_id": unit.id, "provisional": True}]

    r2 = cache.get("beta value one")                    # second refusal -> lossy threshold
    assert cache.report_refusal(r2.read_id) is not None
    assert unit.lossy is True and len(unit.query_keys) == 2

    # UN-RIGGED: the repair extraction is NON-verbatim (real LLMs never reproduce the span
    # byte-for-byte) — the transfer must come from the EXPLICIT keyed-span promotion.
    synth.claims = ["beta shipping rose overall"]
    cache.get("beta kappa")                             # touch -> append-only repair
    assert unit.understanding["claims"] == [
        "alpha beta value", "beta shipping rose overall", missed]   # promoted VERBATIM
    assert all(k.claim_idx == 2 for k in unit.query_keys)   # keys TRANSFERRED to the claim row

    # The measured payoff: the original paraphrase now retrieves the promoted claim TOP.
    r4 = cache.get("beta value")                        # key sim 1.0 >= key_floor
    assert r4.cache_hit is True
    assert r4.pool[0].claim == missed
    assert r4.confidence == pytest.approx(1.0)
    fired = [e for e in events if e["event"] == "key_fired"]
    assert fired and fired[-1]["unit_id"] == unit.id
    assert fired[-1]["key_sim"] == pytest.approx(1.0)


def test_span_key_fires_without_repair() -> None:
    # The regression repro, un-rigged: a behaviorally-earned key is claim_idx=-1 BY
    # CONSTRUCTION (spans are captured because their facts are absent from every claim).
    # It must fire anyway, serving its own span text — no repair required, no verbatim
    # extraction assumed.
    events: list[dict] = []
    ret = _WordRetriever()
    ret.set("src:a", "Alpha beta value coverage improved. "
                     "Beta shipments hit 900 units in kappa region.")
    missed = "Beta shipments hit 900 units in kappa region."
    cache = _pool(ret, _ClaimSynth(["alpha beta value"]),   # synth NEVER emits the span
                  on_event=events.append, query_keys=True)
    cache.get("alpha beta value")                       # build
    unit = next(iter(cache._units.values()))
    r1 = cache.get("beta value")
    assert cache.report_refusal(r1.read_id) is not None
    cache.report_success(r1.read_id)                    # earned: refusal -> success
    assert unit.query_keys[0].claim_idx == -1           # unbound — the normal earned state

    r2 = cache.get("beta value")                        # the IDENTICAL original query
    fired = [e for e in events if e["event"] == "key_fired"]
    assert fired and fired[-1]["key_sim"] == pytest.approx(1.0)
    assert r2.pool[0].claim == missed                   # the span TEXT serves, top-ranked
    assert r2.confidence == pytest.approx(1.0)
    # ...through the normal render, under the F1 default attribution header (§7.1 fork:
    # "## " + the unit's build query, since the unit carries query text).
    assert r2.context["pool"].startswith("## alpha beta value\n- " + missed)
    assert r2.context["pool"].count(missed) == 1        # side channel skips the served copy


def test_repair_promotes_keyed_span() -> None:
    events: list[dict] = []
    ret = _WordRetriever()
    ret.set("src:a", "Alpha beta value coverage improved. "
                     "Beta shipments hit 900 units in kappa region.")
    missed = "Beta shipments hit 900 units in kappa region."
    synth = _ClaimSynth(["alpha beta value"])
    cache = _pool(ret, synth, on_event=events.append, query_keys=True)
    cache.get("alpha beta value")
    unit = next(iter(cache._units.values()))
    for q in ("beta value", "beta value one"):          # two refusals -> lossy
        r = cache.get(q)
        assert cache.report_refusal(r.read_id) is not None
    assert unit.lossy is True
    synth.claims = ["beta shipping rose overall"]       # NON-verbatim repair extraction
    cache.get("beta kappa")                             # the repair touch
    # The keyed span was promoted VERBATIM into the merged claims (mechanism 2)...
    assert missed in unit.understanding["claims"]
    assert all(k.claim_idx >= 0 for k in unit.query_keys)   # ...healing the binding
    # ...so the key now serves the CLAIM row and the span-level row self-retired.
    rows, _m = cache._query_key_rows("")
    assert rows and all(idx >= 0 for _uid, idx, _text, _k in rows)
    r_after = cache.get("beta value")
    assert r_after.pool[0].claim == missed
    assert [e for e in events if e["event"] == "key_fired"]


def test_multiple_span_keys_no_collision() -> None:
    events: list[dict] = []
    cache = _key_cache(on_event=events.append)
    span_a = "Beta shipments hit 900 units in kappa region."
    span_b = "Gamma output fell 5 percent in kappa zone."
    unit = _mk_unit("cog:k2", "alpha value", ["junk"])
    unit.query_keys = (
        QueryKey(embedding=tuple(_embed("kappa")), claim_idx=-1, span_text=span_a),
        QueryKey(embedding=tuple(_embed("kappa")), claim_idx=-1, span_text=span_b),
    )
    cache._units[unit.id] = unit
    r = cache.get("kappa")                              # both keys at sim 1.0 >= floor
    served = [p.claim for p in r.pool]
    assert served.count(span_a) == 1 and served.count(span_b) == 1   # BOTH, no collision
    assert len([e for e in events if e["event"] == "key_fired"]) == 2


def test_serde_span_key_fires_after_load() -> None:
    from coalent.semantic import SQLiteCognitionStore

    store = SQLiteCognitionStore(":memory:")
    ret = _WordRetriever()
    ret.set("src:a", "Alpha beta value coverage improved. "
                     "Beta shipments hit 900 units in kappa region.")
    missed = "Beta shipments hit 900 units in kappa region."
    warm = _pool(ret, _ClaimSynth(["alpha beta value"]), query_keys=True, store=store)
    warm.get("alpha beta value")
    r1 = warm.get("beta value")
    assert warm.report_refusal(r1.read_id) is not None
    warm.report_success(r1.read_id)                     # confirmed -> persisted (idx -1)

    ret2 = _WordRetriever()
    ret2.set("src:a", "Alpha beta value coverage improved. "
                      "Beta shipments hit 900 units in kappa region.")
    events: list[dict] = []
    cold = _pool(ret2, _ClaimSynth(["alpha beta value"]), query_keys=True,
                 store=store, on_event=events.append)
    unit = next(iter(cold._units.values()))
    assert unit.query_keys and unit.query_keys[0].claim_idx == -1   # loaded as stored
    r2 = cold.get("beta value")                         # fires FROM the loaded state
    assert r2.pool[0].claim == missed
    assert [e for e in events if e["event"] == "key_fired"]


def test_query_keys_requires_pool_path() -> None:
    ret = _WordRetriever()
    with pytest.raises(ValueError, match="read_path='pool'"):
        SemanticCache(ret, _ClaimSynth(["a"]),                     # type: ignore[arg-type]
                      embedder=FunctionEmbedder(_embed), query_keys=True,
                      read_path="unit")
    # Explicit unit with the knob OFF never triggers the guard (v0.7: the bare default
    # resolves to pool under a semantic embedder, so the unit leg is pinned explicitly).
    SemanticCache(ret, _ClaimSynth(["a"]),                         # type: ignore[arg-type]
                  embedder=FunctionEmbedder(_embed), read_path="unit")
    _pool(ret, _ClaimSynth(["a"]), query_keys=True)


def test_report_success_confirms_keys() -> None:
    events: list[dict] = []
    ret = _WordRetriever()
    ret.set("src:a", "Alpha beta value coverage improved. "
                     "Beta shipments hit 900 units in kappa region.")
    cache = _pool(ret, _ClaimSynth(["alpha beta value"]),
                  on_event=events.append, query_keys=True)
    r0 = cache.get("alpha beta value")
    unit = cache._units[r0.unit_id]
    r1 = cache.get("beta value")
    assert cache.report_refusal(r1.read_id) is not None
    assert unit.query_keys[0].read_id == r1.read_id     # provisional

    cache.report_success(r1.read_id)                    # the retry worked: confirm
    assert unit.query_keys[0].read_id == ""             # durable now
    confirmed = [e for e in events if e["event"] == "key_confirmed"]
    assert confirmed == [{"event": "key_confirmed", "ts": confirmed[0]["ts"],
                          "unit_id": unit.id}]
    cache.report_success(r1.read_id)                    # idempotent: no second event
    cache.report_success("bogus")                       # unknown id: silent no-op
    assert len([e for e in events if e["event"] == "key_confirmed"]) == 1

    for _ in range(70):                                 # roll the whole ring
        cache.get("alpha beta value")
    assert len(unit.query_keys) == 1                    # confirmed keys DO NOT expire


def test_unconfirmed_keys_expire_with_ring() -> None:
    ret = _WordRetriever()
    ret.set("src:a", "Alpha beta value coverage improved. "
                     "Beta shipments hit 900 units in kappa region.")
    cache = _pool(ret, _ClaimSynth(["alpha beta value"]), query_keys=True)
    r0 = cache.get("alpha beta value")
    unit = cache._units[r0.unit_id]
    r1 = cache.get("beta value")
    assert cache.report_refusal(r1.read_id) is not None
    assert len(unit.query_keys) == 1                    # provisional, never confirmed
    for _ in range(70):                                 # r1 ages out of the 64-read ring
        cache.get("alpha beta value")
    assert unit.query_keys == ()                        # expired with its read


def test_key_fires_only_above_floor() -> None:
    events: list[dict] = []
    cache = _key_cache(on_event=events.append)
    unit = _mk_unit("cog:k", "alpha value", ["alpha value", "gamma three"])
    unit.query_keys = (QueryKey(embedding=tuple(_embed("kappa")), claim_idx=1,
                                span_text="gamma three"),)
    cache._units[unit.id] = unit

    r_low = cache.get("q084")                           # key sim 0.84 < floor 0.85
    assert r_low.confidence == pytest.approx(0.0)       # ignored ENTIRELY: content-only
    assert not [e for e in events if e["event"] == "key_fired"]
    assert unit.query_keys[0].hits == 0

    r_hi = cache.get("q086")                            # key sim 0.86 >= floor
    assert r_hi.cache_hit is True
    assert r_hi.confidence == pytest.approx(0.86)       # the key row carried the gate
    assert r_hi.pool[0].claim == "gamma three"
    fired = [e for e in events if e["event"] == "key_fired"]
    assert len(fired) == 1                              # once per key per read
    assert fired[0]["unit_id"] == unit.id
    assert fired[0]["key_sim"] == pytest.approx(0.86)
    assert unit.query_keys[0].hits == 1


def test_key_never_displaces_gold_wrong_adoption() -> None:
    # STRICT: gold claim X answers by CONTENT; a key on claim Y at 0.86 must never push X
    # out of the serve pack — the overlay adds/raises Y's row, it cannot displace X's.
    cache = _key_cache()
    gold = _mk_unit("cog:gold", "alpha value", ["alpha value"])
    keyed = _mk_unit("cog:keyed", "gamma three", ["gamma three"])
    keyed.query_keys = (QueryKey(embedding=tuple(_embed("kappa")), claim_idx=0,
                                 span_text="gamma three"),)
    cache._units[gold.id] = gold
    cache._units[keyed.id] = keyed

    r = cache.get("qmix")                               # gold content 0.51; Y's key 0.86
    served = [p.claim for p in r.pool]
    assert "alpha value" in served                      # X STILL SERVES
    assert served[0] == "gamma three"                   # the key row ranks by its sim...
    gold_row = next(p for p in r.pool if p.claim == "alpha value")
    assert gold_row.score == pytest.approx((1 - 0.86 ** 2) ** 0.5)   # ...X's score untouched


def test_keys_capped_and_evicted() -> None:
    events: list[dict] = []
    cache = _key_cache(on_event=events.append)
    unit = _mk_unit("cog:full", "alpha value", ["alpha value"])
    hits = [5, 1, 3, 4, 2, 6, 7, 8]
    unit.query_keys = tuple(
        QueryKey(embedding=tuple(_embed("kappa")), claim_idx=0,
                 span_text=f"old-{i}", hits=h)
        for i, h in enumerate(hits))
    cache._units[unit.id] = unit

    cache._attach_key(unit, tuple(_embed("beta")), "newcomer", "read-x")
    texts = [k.span_text for k in unit.query_keys]
    assert len(texts) == 8                              # capped at 8 keys/unit
    assert "newcomer" in texts                          # the newcomer always lands
    assert "old-1" not in texts                         # the lowest-hits EXISTING key evicted
    assert set(texts) == {"old-0", "old-2", "old-3", "old-4",
                          "old-5", "old-6", "old-7", "newcomer"}
    assert [e for e in events if e["event"] == "key_attached"]


def test_key_serde_roundtrip() -> None:
    confirmed = QueryKey(embedding=(0.1, 0.2), claim_idx=1, span_text="fact", hits=3)
    provisional = QueryKey(embedding=(0.3, 0.4), claim_idx=-1, span_text="other",
                           hits=0, read_id="read-9")
    unit = Cognition(
        id="cog:x", namespace="", query="alpha", query_embedding=(1.0, 0.0),
        understanding={"claims": ["a", "fact"]}, evidence=(),
        provenance=ProvenanceManifest("s", "p"),
        query_keys=(confirmed, provisional),
    )
    data = json.loads(cognition_to_json(unit))
    # CONFIRMED keys persist (embedding included — a key has no text to regenerate from);
    # provisional keys are process-local and never written.
    assert data["query_keys"] == [{"embedding": [0.1, 0.2], "claim_idx": 1,
                                   "span_text": "fact", "hits": 3}]
    restored = cognition_from_dict(data)
    assert restored.query_keys == (confirmed,)
    data.pop("query_keys")                              # an old dict loads fine
    assert cognition_from_dict(data).query_keys == ()
    # A keyless unit writes NO query_keys key at all (lean, byte-identical v0.5 JSON).
    bare = cognition_from_dict(data)
    assert "query_keys" not in cognition_to_dict(bare)


def test_query_keys_off_byte_identical() -> None:
    def run(**kw: object) -> tuple[SemanticCache, list[dict], list[object], str | None]:
        events: list[dict] = []
        ret = _WordRetriever()
        ret.set("src:a", "Alpha beta value coverage improved. "
                         "Beta shipments hit 900 units in kappa region.")
        cache = _pool(ret, _ClaimSynth(["alpha beta value"]), on_event=events.append,
                      clock=lambda: 1000.0, **kw)       # residual_spans ON; keys per kw
        r0 = cache.get("alpha beta value")
        r1 = cache.get("beta value")
        payload = cache.report_refusal(r1.read_id)      # the attach point — keys OFF here
        r2 = cache.get("beta value one")
        return cache, events, [r0, r1, r2], payload

    default_cache, default_events, default_results, default_payload = run()
    explicit_cache, explicit_events, explicit_results, explicit_payload = run(query_keys=False)
    assert default_results == explicit_results
    assert default_events == explicit_events
    assert default_payload == explicit_payload
    kinds = {e["event"] for e in default_events}
    assert not kinds & {"key_attached", "key_confirmed", "key_fired"}
    for unit in default_cache._units.values():
        assert unit.query_keys == ()                    # no key rows exist anywhere
        assert "query_keys" not in cognition_to_dict(unit)
    default_cache.report_success(default_results[1].read_id)   # inert: no-op, no events
    assert not [e for e in default_events if e["event"] == "key_confirmed"]


# ------------------------------------------------------------------ default-OFF pin

def test_residual_spans_off_by_default_byte_identical() -> None:
    def run(**kw: object) -> tuple[SemanticCache, list[dict], list[object]]:
        events: list[dict] = []
        ret = _WordRetriever()
        ret.set("src:a", "The alpha value figure was 42 in June.")
        cache = SemanticCache(ret, _ClaimSynth(["alpha value"]),  # type: ignore[arg-type]
                              embedder=FunctionEmbedder(_embed), read_path="pool",
                              serve_gate=0.5, coverage_floor=0.9, clock=lambda: 1000.0,
                              on_event=events.append, **kw)       # type: ignore[arg-type]
        results = [cache.get(q)
                   for q in ("alpha value", "alpha two", "alpha value", "beta junk")]
        return cache, events, results

    default_cache, default_events, default_results = run()
    explicit_cache, explicit_events, explicit_results = run(residual_spans=False)
    # Build + serve + escalation + cold miss: byte-identical Results, events, and stats.
    assert default_results == explicit_results
    assert default_events == explicit_events
    assert default_cache.stats() == explicit_cache.stats()
    # None of the tier-2 machinery ran, nothing was captured, nothing new is persisted.
    kinds = {e["event"] for e in default_events}
    assert not kinds & {"residual_served", "residual_fallback", "unit_marked_lossy",
                        "unit_repaired"}
    for unit in default_cache._units.values():
        assert unit.residual_spans == ()
        assert unit.span_hits == 0 and unit.lossy is False
        assert not {"residual_spans", "span_hits", "lossy"} & set(cognition_to_dict(unit))
    # The behavioral channel is inert when OFF: a real read_id yields None (nothing logged).
    assert default_results[0].read_id == "read-1"       # ids stay deterministic either way
    assert default_cache.report_refusal(default_results[0].read_id) is None
