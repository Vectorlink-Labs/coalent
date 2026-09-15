"""v0.5 — workload presets + the bridge restart, logic tier (hermetic, deterministic).

The multi_hop preset exists because the headline capability was DORMANT out of the box:
recall_threshold defaulted to coverage_floor, far below the 0.70 fewest-misses operating
point, and the bridge restart lived only in bench_multihop. These tests pin (a) the preset
plumbing (explicit kwargs always win; unknown preset is a loud error; "default" is a no-op)
and (b) the bridge mechanics on the user's own motivating example:

    unit A: "john works payments"       (people — different team's source)
    unit B: "payments owns invoicing"   (services — another team's source)
    Q: "john deliverable?" -> matches A, under-covers, hop-2 shares NOTHING with the
    question — only the bridge entity ("payments") in A's own claims can reach B.

Embedding QUALITY (does the bridge lift real multi-hop accuracy) was validated on real
OpenAI in bench_multihop (support-recall 92%, answer acc 0% -> 100%); real-document
validation lands in M4. Here we pin the algorithm so it can never silently break.
"""
from __future__ import annotations

import pytest

from coalent import PRESETS, FunctionEmbedder
from coalent.semantic import Chunk, InMemoryRetriever, SemanticCache, Synthesis

# Each content word is its own meaning axis -> exact, hand-checkable cosines.
_AXES = ("john", "bob", "works", "payments", "hr", "owns", "invoicing",
         "recruiting", "office", "paris", "deliverable")


def _embed(text: str) -> list[float]:
    words = set(text.lower().replace("?", " ").replace(";", " ").split())
    v = [1.0 if a in words else 0.0 for a in _AXES]
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v] if norm else v


class _AtomSynth:
    """Splits each chunk into atomic '; '-separated claims — a controllable stand-in for
    an LLM that emits atomic, source-grounded claims."""

    def synthesize(self, query: str, chunks: list[Chunk]) -> Synthesis:
        text = " ".join(c.text for c in chunks)
        claims = [s.strip() for s in text.split(";") if s.strip()]
        return Synthesis(understanding={"summary": text, "claims": claims},
                         used=list(range(len(chunks))))


def _world(**kw: object) -> tuple[SemanticCache, dict[str, str]]:
    """Three teams' sources, one warm unit each: people (A), services (B), offices (C).
    C exists to CROWD OUT the bridge fact from plain query-ranked recall (recall_limit=2):
    its 'john office paris' claim shares the query's 'john' axis; B's claim shares none."""
    retriever = InMemoryRetriever(top_k=1)
    retriever.add("people", "john works payments; bob works hr")
    retriever.add("services", "payments owns invoicing; hr owns recruiting")
    retriever.add("offices", "john office paris; bob office paris")
    cache = SemanticCache(
        retriever,
        _AtomSynth(),
        embedder=FunctionEmbedder(_embed),
        hit_threshold=0.30,
        coverage_floor=0.35,
        recall_limit=2,
        read_path="unit",
        **kw,  # type: ignore[arg-type]
    )
    ids: dict[str, str] = {}
    for name, seed in (("people", "john bob works"), ("services", "owns invoicing recruiting"),
                       ("offices", "office paris")):
        ids[name] = cache.get(seed).unit_id
    return cache, ids


# ------------------------------------------------------------- preset plumbing
def test_preset_multi_hop_arms_the_validated_knobs() -> None:
    cache, _ = _world(preset="multi_hop")
    assert cache._recall_threshold == 0.7
    assert cache._recall_bridge is True


def test_no_preset_leaves_v04_behavior() -> None:
    cache, _ = _world()
    assert cache._recall_threshold is None      # inherits coverage_floor, the v0.4 default
    assert cache._recall_bridge is False


def test_default_preset_is_a_noop() -> None:
    plain, _ = _world()
    default, _ = _world(preset="default")
    for attr in ("_recall_threshold", "_recall_bridge", "_bridge_limit",
                 "_recall_limit", "_threshold", "_coverage_floor"):
        assert getattr(default, attr) == getattr(plain, attr)


def test_unknown_preset_is_a_loud_error() -> None:
    with pytest.raises(ValueError, match="unknown preset .*multihop.*available.*multi_hop"):
        _world(preset="multihop")


def test_explicit_kwargs_beat_the_preset() -> None:
    cache, _ = _world(preset="multi_hop", recall_threshold=0.55, recall_bridge=False)
    assert cache._recall_threshold == 0.55
    assert cache._recall_bridge is False


def test_presets_registry_is_exported() -> None:
    assert set(PRESETS) >= {"default", "multi_hop"}
    assert PRESETS["multi_hop"]["recall_threshold"] == 0.7


# --------------------------------------------------- the headline: bridge mechanics
def test_dormant_by_default_fires_with_preset() -> None:
    """The user's motivating example: 'what is john's deliverable?' — hop-2 (payments owns
    invoicing) shares NO axis with the question. v0.4 defaults: recall never fires (coverage
    cos({john,deliverable},"john works payments") = 1/sqrt(6) ~ 0.408 sits above the
    inherited 0.35 trigger). multi_hop preset: recall fires at 0.70, query-ranked recall is
    crowded out by the 'john office paris' distractor, and the BRIDGE (via 'payments' in A's
    own claims) is the only path that serves the hop-2 fact."""
    plain, _ = _world()
    r_plain = plain.get("john deliverable")
    assert r_plain.recalled == []                                  # dormant: the v0.4 story

    armed, ids = _world(preset="multi_hop")
    r = armed.get("john deliverable")
    assert r.cache_hit and r.unit_id == ids["people"]              # hop-1 matched
    claims = [rc.claim for rc in r.recalled]
    assert "payments owns invoicing" in claims                     # hop-2 SERVED
    bridge = next(rc for rc in r.recalled if rc.claim == "payments owns invoicing")
    assert bridge.unit_id == ids["services"]                       # provenance intact
    served = r.context.get("understanding", {})
    assert "payments owns invoicing" in served.get("recalled_claims", [])   # in the SERVED payload


def test_bridge_never_inflates_coverage() -> None:
    """Bridge scores are bridge-similarity, not query-similarity: serving them must not
    raise the coverage the escalation gate judges — else a bridged read could dodge the
    RAG floor on the strength of a claim that does NOT answer the query."""
    armed, _ = _world(preset="multi_hop")
    r = armed.get("john deliverable")
    # query-coverage: cos({john,deliverable}, "john works payments") = 1/sqrt(6) ~ 0.408;
    # every query-similar claim scores the same or less -> coverage stays well under 0.7.
    assert r.coverage < 0.7
    assert [rc for rc in r.recalled if rc.claim == "payments owns invoicing"]
    # the decisive assertion: coverage equals the best QUERY-similar evidence exactly,
    # unmoved by the bridge claim's own (bridge-axis) score
    query_scores = [rc.score for rc in r.recalled if rc.claim != "payments owns invoicing"]
    assert query_scores and abs(r.coverage - max(query_scores)) < 1e-9


def test_bridge_expands_to_new_units_only() -> None:
    armed, ids = _world(preset="multi_hop")
    r = armed.get("john deliverable")
    bridged = [rc for rc in r.recalled if rc.claim == "payments owns invoicing"]
    assert bridged and bridged[0].unit_id not in {ids["people"]}   # never the matched unit
    # and never a duplicate of a claim already recalled by query-similarity
    assert len({rc.claim for rc in r.recalled}) == len(r.recalled)


def test_single_hop_served_context_is_identical_with_preset() -> None:
    """A fully-covered single-hop read must be byte-identical with and without the preset —
    the covenant that upgrading (or opting in) never changes the common path."""
    plain, _ = _world()
    armed, _ = _world(preset="multi_hop")
    q = "john works payments"                    # fully covered by unit A's own claim
    r_plain, r_armed = plain.get(q), armed.get(q)
    assert r_plain.recalled == [] and r_armed.recalled == []       # recall never fired
    assert r_armed.context == r_plain.context                      # served context identical
    assert r_armed.understanding == r_plain.understanding
