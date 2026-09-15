"""v0.5 — the M4-driven production fixes, pinned (hermetic).

(1) adaptive_hit: the M4 warm-up experiment proved match scores INFLATE as units accumulate,
so a fixed hit_threshold sinks under the noise floor and the cache absorbs everything (null-hit
0%->100% at the default). The adaptive gate self-calibrates against the cache's own cross-unit
score distribution. (2) split_by_artifact: retrieval mixing chunks from several sources must
never blend into one lossy understanding (the digest/multi-source finding)."""
from __future__ import annotations

from coalent import FunctionEmbedder
from coalent.semantic import Chunk, InMemoryRetriever, SemanticCache, Synthesis

_AXES = ("common", "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta",
         "theta", "iota", "kappa", "target", "unique")


def _embed(text: str) -> list[float]:
    words = set(text.lower().split())
    v = [1.0 if a in words else 0.0 for a in _AXES]
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v] if n else v


class _Synth:
    def synthesize(self, query: str, chunks: list[Chunk]) -> Synthesis:
        text = " ".join(c.text for c in chunks)
        return Synthesis(understanding={"summary": text, "claims": [text]},
                         used=list(range(len(chunks))))


def _crowded_cache(adaptive: bool) -> SemanticCache:
    """Ten units all sharing the 'common' axis — cross-unit best-match scores run high,
    exactly the score-inflation regime the warm-up experiment measured."""
    retriever = InMemoryRetriever()
    cache = SemanticCache(retriever, _Synth(), embedder=FunctionEmbedder(_embed),
                          hit_threshold=0.30, coverage_floor=0.0, adaptive_hit=adaptive,
                          read_path="unit")
    topics = _AXES[1:11]
    for t in topics:
        retriever.add(f"src:{t}", f"common {t} fact")
    bar = cache._threshold
    cache._threshold = 2.0
    for t in topics:
        cache.get(f"common {t}")
    cache._threshold = bar
    return cache


def test_fixed_threshold_absorbs_offtopic_but_adaptive_builds() -> None:
    fixed = _crowded_cache(adaptive=False)
    r = fixed.get("common unique")                 # off-topic apart from the shared word
    assert r.cache_hit is True                     # the false-alarm regime: absorbed

    adaptive = _crowded_cache(adaptive=True)
    n_before = len(adaptive._units)
    r2 = adaptive.get("common unique")
    assert r2.cache_hit is False                   # the bar rose above the noise ceiling
    assert len(adaptive._units) == n_before + 1    # -> a REAL build
    assert adaptive._noise_ceiling > 0.30          # calibrated above the base threshold


def test_adaptive_still_hits_genuine_matches() -> None:
    adaptive = _crowded_cache(adaptive=True)
    r = adaptive.get("common alpha")               # a genuinely cached topic
    assert r.cache_hit is True                     # strong matches clear the raised bar


def test_split_by_artifact_builds_one_unit_per_source() -> None:
    class MixedRetriever:
        def retrieve(self, query: str, namespace: str | None = None) -> list[Chunk]:
            return [Chunk(artifact_id="art:A", text="alpha beta fact one"),
                    Chunk(artifact_id="art:A", text="alpha gamma fact two"),
                    Chunk(artifact_id="art:B", text="delta epsilon other topic")]

    cache = SemanticCache(MixedRetriever(), _Synth(), embedder=FunctionEmbedder(_embed),
                          hit_threshold=0.99, coverage_floor=0.0, split_by_artifact=True,
                          read_path="unit")
    cache.get("alpha beta")
    units = list(cache._units.values())
    assert len(units) == 2                         # dominant art:A unit + art:B sibling
    per_unit_arts = [set(c.artifact_id for c in u.evidence) for u in units]
    assert all(len(a) == 1 for a in per_unit_arts)  # NO unit blends sources
    assert {"art:A"} in per_unit_arts and {"art:B"} in per_unit_arts


def test_split_off_is_v04_behavior() -> None:
    class MixedRetriever:
        def retrieve(self, query: str, namespace: str | None = None) -> list[Chunk]:
            return [Chunk(artifact_id="art:A", text="alpha fact"),
                    Chunk(artifact_id="art:B", text="delta fact")]

    cache = SemanticCache(MixedRetriever(), _Synth(), embedder=FunctionEmbedder(_embed),
                          hit_threshold=0.99, coverage_floor=0.0, read_path="unit")
    cache.get("alpha")
    assert len(cache._units) == 1                  # default: one blended unit, unchanged


def test_adaptive_gate_never_blocks_reuse() -> None:
    """Gate v2: the noise bar governs NOVEL queries only — an exact/near revisit rides the
    seed channel and always hits (the M4 replay finding: v1 built on 357/360 reads)."""
    adaptive = _crowded_cache(adaptive=True)
    n = len(adaptive._units)
    r = adaptive.get("common alpha")                  # exact revisit of a cached seed query
    assert r.cache_hit is True
    assert len(adaptive._units) == n                  # no rebuild: reuse preserved


def test_provenance_admission_prevents_duplicate_understanding() -> None:
    """The sick-leave case: a paraphrase question over an ALREADY-UNDERSTOOD source must hit
    the existing unit, not mint a duplicate (the GraphRAG understanding-tax)."""
    class HRRetriever:
        def retrieve(self, query: str, namespace: str | None = None):
            return [Chunk(artifact_id="hr:leave",
                          text="alpha sick leave policy beta medical certificate gamma")]

    cache = SemanticCache(HRRetriever(), _Synth(), embedder=FunctionEmbedder(_embed),
                          hit_threshold=0.99,            # score gate forced to MISS
                          coverage_floor=0.0, provenance_admission=True, read_path="unit")
    r1 = cache.get("alpha beta")                          # first question -> builds
    assert r1.cache_hit is False and len(cache._units) == 1
    r2 = cache.get("gamma delta epsilon")                 # different question, same source
    assert r2.cache_hit is True                           # hit-by-provenance
    assert len(cache._units) == 1                         # NO duplicate unit
    assert r2.unit_id == r1.unit_id


class WidenRetriever:
    """Six-chunk source; query retrieval surfaces only 2 chunks (the keyhole)."""

    CHUNKS = [f"alpha part{i} beta fact{i}" for i in range(6)]

    def retrieve(self, query: str, namespace: str | None = None) -> list[Chunk]:
        return [Chunk(artifact_id="doc:x", text=self.CHUNKS[0]),
                Chunk(artifact_id="doc:x", text=self.CHUNKS[1])]

    def widen(self, artifact_id: str, *, limit: int | None = None,
              namespace: str | None = None) -> list[Chunk]:
        assert artifact_id == "doc:x"
        chunks = [Chunk(artifact_id="doc:x", text=t) for t in self.CHUNKS]
        return chunks[:limit] if limit else chunks


def test_widening_builds_from_the_source_not_the_keyhole() -> None:
    cache = SemanticCache(WidenRetriever(), _Synth(), embedder=FunctionEmbedder(_embed),
                          hit_threshold=0.99, coverage_floor=0.0,
                          split_by_artifact=True, widen_chunks=24, read_path="unit")
    r = cache.get("alpha beta")
    unit = cache._units[r.unit_id]
    assert len(unit.evidence) == 6                    # the whole source, not 2 chunks
    assert "fact5" in str(unit.understanding)         # tail content captured


def test_widening_off_is_v04_behavior() -> None:
    cache = SemanticCache(WidenRetriever(), _Synth(), embedder=FunctionEmbedder(_embed),
                          hit_threshold=0.99, coverage_floor=0.0, split_by_artifact=True,
                          read_path="unit")
    r = cache.get("alpha beta")
    assert len(cache._units[r.unit_id].evidence) == 2  # keyhole preserved when OFF


def test_containment_admission_rebuilds_thin_units_widened() -> None:
    """The 96%% blind-spot fix: a unit that never read the probed chunks must NOT satisfy
    admission; with widen_on_admission the thin unit is rebuilt widened IN PLACE."""

    class TwoPhase(WidenRetriever):
        def __init__(self) -> None:
            self.phase = 0

        def retrieve(self, query: str, namespace: str | None = None) -> list[Chunk]:
            if self.phase == 0:                       # first build: keyhole chunks 0-1
                return [Chunk(artifact_id="doc:x", text=self.CHUNKS[0]),
                        Chunk(artifact_id="doc:x", text=self.CHUNKS[1])]
            return [Chunk(artifact_id="doc:x", text=self.CHUNKS[4]),  # probe: unseen chunks
                    Chunk(artifact_id="doc:x", text=self.CHUNKS[5])]

    retriever = TwoPhase()
    cache = SemanticCache(retriever, _Synth(), embedder=FunctionEmbedder(_embed),
                          hit_threshold=0.99, coverage_floor=0.0, split_by_artifact=True,
                          provenance_admission=True, widen_chunks=24, read_path="unit")
    cache.get("alpha beta")                           # keyhole? no — widening ON -> full
    # force a THIN unit to exercise the predicate: rebuild world with widening off first
    retriever2 = TwoPhase()
    thin = SemanticCache(retriever2, _Synth(), embedder=FunctionEmbedder(_embed),
                         hit_threshold=0.99, coverage_floor=0.0, split_by_artifact=True,
                         provenance_admission=True, widen_chunks=24, read_path="unit",
                         source_fetcher=lambda a: [Chunk(artifact_id=a, text=t)
                                                   for t in WidenRetriever.CHUNKS])
    thin._widen_chunks = None                         # phase 0 builds a KEYHOLE unit
    r1 = thin.get("alpha beta")
    assert len(thin._units[r1.unit_id].evidence) == 2
    thin._widen_chunks = 24                           # widening now available
    retriever2.phase = 1
    r2 = thin.get("beta fact5 gamma")                 # probes chunks the unit never read
    assert len(thin._units) == 1                      # NO duplicate unit
    assert len(thin._units[r2.unit_id].evidence) == 6  # rebuilt widened in place
