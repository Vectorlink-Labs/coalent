"""Phase B — semantic per-claim coverage + escalation.

Uses a tiny deterministic "axis" embedder (each topic word is an orthogonal axis) so
cosines are exact and the semantic behavior is testable without a live model: a query
on the unit's topic HITS, but if no single claim covers it the hit escalates to the
retained raw (restoring the RAG floor — semantically, not lexically).
"""
from __future__ import annotations

from coalent import FunctionEmbedder
from coalent.semantic import Chunk, ContextStrategy, InMemoryRetriever, SemanticCache, Synthesis

_AXES = ("card", "stolen", "expire", "fee", "leave", "vacation")


def _axis_embed(text: str) -> list[float]:
    t = text.lower()
    v = [1.0 if axis in t else 0.0 for axis in _AXES]
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v] if norm else v


class _LossyCardSynth:
    """Digest that captures expire/fee but DROPS the stolen-card fact (which the raw
    still holds) — the lossy-digest case the coverage gate must catch."""

    def synthesize(self, query: str, chunks: list[Chunk]) -> Synthesis:
        return Synthesis(
            understanding={"summary": "card expire fee", "claims": ["card expire", "card fee"]},
            used=list(range(len(chunks))),
        )


def _cache(**kw: object) -> SemanticCache:
    retriever = InMemoryRetriever()
    retriever.add("doc:cards", "lost or stolen cards can be replaced with the order number")
    defaults: dict[str, object] = dict(hit_threshold=0.4, coverage_floor=0.6)
    defaults.update(kw)
    return SemanticCache(
        retriever, _LossyCardSynth(), embedder=FunctionEmbedder(_axis_embed), **defaults
    )  # type: ignore[arg-type]


def test_under_covering_hit_escalates_to_retained_raw() -> None:
    cache = _cache()
    cache.get("card")  # cold build: claims about expire/fee, raw retains 'stolen'

    result = cache.get("stolen card")  # on-topic (hits) but no claim covers 'stolen'
    assert result.cache_hit is True       # reused the unit (still a hit)
    assert result.escalated is True       # ...but escalated: no claim covered it
    assert "stolen" in " ".join(result.context["raw"])  # fresh raw surfaced for the answer


def test_well_covered_hit_does_not_escalate() -> None:
    cache = _cache()
    cache.get("card")

    result = cache.get("card expire")  # a claim covers this exactly
    assert result.cache_hit is True
    assert result.escalated is False
    assert result.context["raw"] == []  # understanding-only, no raw needed


def test_escalation_can_be_disabled() -> None:
    cache = _cache(enable_coverage_escalation=False)
    cache.get("card")

    result = cache.get("stolen card")  # under-covers, but escalation is OFF
    assert result.cache_hit is True
    assert result.escalated is False
    assert result.context["raw"] == []


def test_stats_report_escalation_rate() -> None:
    cache = _cache()
    cache.get("card")          # miss (build)
    cache.get("card expire")   # hit, well covered
    cache.get("stolen card")   # hit, escalates

    s = cache.stats()
    assert s["hits"] == 2
    assert s["escalations"] == 1
    assert s["escalation_rate"] == 0.5  # 1 of 2 hits escalated


def test_context_raw_strategy_does_not_escalate() -> None:
    # CONTEXT_RAW already ships raw, so an under-covering hit needs no extra fetch.
    cache = _cache(strategy=ContextStrategy.CONTEXT_RAW)
    cache.get("card")

    result = cache.get("stolen card")
    assert result.cache_hit is True
    assert result.escalated is False                       # raw already in the payload
    assert "stolen" in " ".join(result.context["raw"])     # and it's there
