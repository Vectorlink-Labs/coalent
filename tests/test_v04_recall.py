"""v0.4 — cross-unit claim recall, logic tier (hermetic, deterministic).

These tests prove the SUBSTRATE mechanics with a controllable synonym-axis embedder:
pooling per-claim memory ACROSS units, MaxSim ranking, de-dup, the semantic-embedder
gate, the tunable recall threshold, and — the headline — recovering a bridge fact that
lives in a unit OTHER than the single best match. Embedding *quality* (does recall help
real multi-hop accuracy) is validated separately on real OpenAI embeddings at the M0 gate;
here we pin the algorithm so it can never silently break.

The toy world is a 2-hop question: "alice capital" needs hop-1 (alice -> france) from the
PEOPLE unit and hop-2 (france -> capital paris) from the GEO unit. Single-unit Coalent can
only serve one; cross-unit recall surfaces both.
"""
from __future__ import annotations

from coalent import FunctionEmbedder
from coalent.semantic import Chunk, InMemoryRetriever, SemanticCache, Synthesis

# Each content word is its own meaning axis -> exact, hand-checkable cosines.
_AXES = ("alice", "bob", "france", "germany", "capital", "paris", "berlin", "bridge")


def _embed(text: str) -> list[float]:
    words = set(text.lower().replace("?", " ").replace(";", " ").split())
    v = [1.0 if a in words else 0.0 for a in _AXES]
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v] if norm else v


class _AtomSynth:
    """Splits each chunk into atomic '; '-separated claims; summary = full text — a
    controllable stand-in for an LLM that emits atomic, source-grounded claims."""

    def synthesize(self, query: str, chunks: list[Chunk]) -> Synthesis:
        text = " ".join(c.text for c in chunks)
        claims = [s.strip() for s in text.split(";") if s.strip()]
        return Synthesis(
            understanding={"summary": text, "claims": claims},
            used=list(range(len(chunks))),
        )


def _world(**kw: object) -> tuple[SemanticCache, str, str]:
    """A warm cache with one unit per document (the realistic steady state)."""
    retriever = InMemoryRetriever(top_k=1)
    retriever.add("people", "alice lives in france; bob lives in germany")
    retriever.add("geo", "france capital paris; germany capital berlin")
    cache = SemanticCache(
        retriever,
        _AtomSynth(),
        embedder=FunctionEmbedder(_embed),
        hit_threshold=0.35,
        coverage_floor=0.40,
        **kw,  # type: ignore[arg-type]
    )
    people = cache.get("alice bob")          # builds the PEOPLE unit
    geo = cache.get("capital paris berlin")  # builds the GEO unit (orthogonal seed)
    return cache, people.unit_id, geo.unit_id


def _qe(text: str) -> tuple[float, ...]:
    return tuple(FunctionEmbedder(_embed).embed(text))


# --------------------------------------------------------------- the headline
def test_recall_surfaces_cross_unit_bridge_fact() -> None:
    """The whole thesis: a fact in a unit OTHER than the best match is recovered."""
    cache, people_id, geo_id = _world(
        cross_unit_recall=True, recall_threshold=0.55, recall_limit=6
    )
    result = cache.get("alice capital")  # 2-hop; best single unit is PEOPLE (has 'alice')

    assert result.cache_hit is True
    assert result.unit_id == people_id                       # single best match == people
    claims = [rc.claim for rc in result.recalled]
    assert "france capital paris" in claims                  # the hop-2 fact...
    sources = {rc.unit_id for rc in result.recalled}
    assert geo_id in sources and len(sources) >= 2           # ...pooled from the GEO unit
    # and it is injected into the context the answerer will actually read
    assert "france capital paris" in result.context["understanding"]["recalled_claims"]


# --------------------------------------------------------------- substrate mechanics
def test_recall_pools_across_units_and_ranks_by_maxsim() -> None:
    cache, _people, _geo = _world(cross_unit_recall=True)
    recalled = cache._recall_claims(_qe("alice capital"), "", limit=4)

    assert len(recalled) <= 4
    scores = [rc.score for rc in recalled]
    assert scores == sorted(scores, reverse=True)            # MaxSim, descending
    assert len({rc.unit_id for rc in recalled}) >= 2         # genuinely cross-unit


def test_recall_respects_limit() -> None:
    cache, *_ = _world(cross_unit_recall=True)
    assert len(cache._recall_claims(_qe("alice capital"), "", limit=2)) == 2


def test_recall_dedups_repeated_claim_text() -> None:
    """A single-claim unit has summary == claim, so claim_texts carries a duplicate —
    the pool must collapse it to one entry (highest score wins)."""
    retriever = InMemoryRetriever(top_k=1)
    retriever.add("d1", "alice france bridge")               # one claim; summary == it
    cache = SemanticCache(
        retriever, _AtomSynth(), embedder=FunctionEmbedder(_embed),
        hit_threshold=0.35, cross_unit_recall=True,
    )
    cache.get("alice france bridge")
    recalled = cache._recall_claims(_qe("alice france"), "", limit=10)

    texts = [rc.claim for rc in recalled]
    assert texts.count("alice france bridge") == 1


def test_recall_excludes_dirty_units() -> None:
    cache, _people, geo_id = _world(cross_unit_recall=True)
    cache.source_changed("geo", text="france capital lyon")  # real change -> dirties GEO
    recalled = cache._recall_claims(_qe("alice capital"), "", limit=10)

    assert all(rc.unit_id != geo_id for rc in recalled)      # stale unit not pooled


# --------------------------------------------------------------- gates & invariants
def test_recall_gated_off_under_hashing_embedder() -> None:
    """Lexical HashingEmbedder => claim cosine is keyword overlap, not meaning => OFF."""
    retriever = InMemoryRetriever(top_k=1)
    retriever.add("people", "alice lives in france; bob lives in germany")
    retriever.add("geo", "france capital paris; germany capital berlin")
    cache = SemanticCache(
        retriever, _AtomSynth(),  # default embedder forced to HashingEmbedder by conftest
        cross_unit_recall=True, recall_threshold=0.9,        # would fire if semantic
    )
    cache.get("alice bob")
    cache.get("capital paris berlin")

    assert cache._is_semantic_embedder() is False
    assert cache.get("alice capital").recalled == []


def test_well_covered_read_is_byte_identical_to_v03() -> None:
    """At/above the recall threshold, recall must not run — single-hop is untouched."""
    on, _p, _g = _world(cross_unit_recall=True, recall_threshold=0.55)
    off, _p2, _g2 = _world(cross_unit_recall=False)          # explicit v0.3 (recall now defaults ON)
    query = "alice france"                                   # fully covered by PEOPLE unit

    r_on, r_off = on.get(query), off.get(query)
    assert r_on.recalled == []
    assert r_on.context == r_off.context
    assert r_on.understanding == r_off.understanding


def test_recall_threshold_controls_the_trigger() -> None:
    """Below threshold recall fires; lower the threshold under the coverage and it stops."""
    fires, *_ = _world(cross_unit_recall=True, recall_threshold=0.55)
    quiet, *_ = _world(cross_unit_recall=True, recall_threshold=0.45)
    # 'alice capital' covers the PEOPLE unit at 0.5: 0.5 < 0.55 fires, 0.5 >= 0.45 does not.
    assert fires.get("alice capital").recalled != []
    assert quiet.get("alice capital").recalled == []


# --------------------------------------------------------------- M2 ordered flow + learn + needs_retrieval
def _flow_cache(**kw: object) -> SemanticCache:
    """Same two-unit world as _world but with the coverage knobs left free to override."""
    retriever = InMemoryRetriever(top_k=1)
    retriever.add("people", "alice lives in france; bob lives in germany")
    retriever.add("geo", "france capital paris; germany capital berlin")
    cache = SemanticCache(
        retriever, _AtomSynth(), embedder=FunctionEmbedder(_embed),
        hit_threshold=0.35, **kw,  # type: ignore[arg-type]
    )
    cache.get("alice bob")              # builds the PEOPLE unit (seed = 'alice bob')
    cache.get("capital paris berlin")   # builds the GEO unit
    return cache


def test_coverage_uses_cross_unit_recall() -> None:
    """The ordered flow re-covers: result.coverage becomes the best claim across ALL units (S1c)."""
    cache = _flow_cache(
        cross_unit_recall=True, recall_threshold=0.99, coverage_floor=0.0,
        enable_coverage_escalation=False,
    )
    result = cache.get("alice capital")
    assert result.recalled                                   # recall fired
    assert abs(result.coverage - max(c.score for c in result.recalled)) < 1e-9


def test_needs_retrieval_tracks_coverage_floor() -> None:
    cache = _flow_cache(coverage_floor=0.9, enable_coverage_escalation=False)
    covered = cache.get("alice lives in france")            # exact claim -> coverage 1.0 >= 0.9
    assert covered.needs_retrieval is False
    under = cache.get("alice bob")                          # coverage ~0.5 < 0.9
    assert under.needs_retrieval is True
    assert under.needs_retrieval is (under.coverage < 0.9)


def test_learn_on_escalation_grows_the_cache() -> None:
    cache = _flow_cache(coverage_floor=0.9, learn_on_escalation=True)
    before = len(cache._units)
    # A DISTINCT query that still hits the people unit (not its exact seed, so a NEW id is minted).
    result = cache.get("alice germany")                    # HIT, coverage 0.5 < 0.9 -> escalate -> LEARN
    assert result.escalated is True
    assert len(cache._units) == before + 1                  # the escalation was cached as a new unit


def test_escalation_without_learn_keeps_cache_size() -> None:
    cache = _flow_cache(coverage_floor=0.9)                 # learn_on_escalation defaults OFF
    before = len(cache._units)
    result = cache.get("alice bob")
    assert result.escalated is True
    assert len(cache._units) == before                      # default: escalate but do NOT learn


def test_cost_sensitive_floor_and_surfaced_thresholds() -> None:
    # tau* = 1 - escalation_cost / miss_cost, clamped to [0, 1]
    assert SemanticCache.cost_sensitive_floor(1.0, 10.0) == 0.9   # costly miss -> high floor
    assert SemanticCache.cost_sensitive_floor(9.0, 10.0) == 0.1   # cheap miss -> low floor
    assert SemanticCache.cost_sensitive_floor(20.0, 10.0) == 0.0  # clamp low
    assert SemanticCache.cost_sensitive_floor(1.0, 0.0) == 0.0    # no miss cost -> never escalate
    # plugs into coverage_floor and is observable via stats()
    cache = _flow_cache(coverage_floor=SemanticCache.cost_sensitive_floor(1.0, 4.0))  # 0.75
    assert cache.stats()["coverage_floor"] == 0.75
