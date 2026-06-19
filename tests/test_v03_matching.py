"""Phase E — semantic matching guarantees, hermetic.

Uses a deterministic *synonym-axis* embedder: synonyms collapse to a shared axis, so
paraphrases are close and different topics are far — modeling meaning without a live
model. Covers the v0.3 wins: paraphrase recall, surface-form false-hit rejection, no
seed-drift, and lazy backfill of legacy (pre-v0.3) units.
"""
from __future__ import annotations

from coalent import FunctionEmbedder
from coalent.semantic import (
    Chunk,
    InMemoryRetriever,
    SemanticCache,
    SQLiteCognitionStore,
    Synthesis,
)

_SYN = {
    "leave": "timeoff", "vacation": "timeoff", "pto": "timeoff", "holiday": "timeoff",
    "days": "timeoff", "entitlement": "timeoff",
    "gift": "giftcard", "card": "giftcard", "cards": "giftcard",
    "return": "returns", "returns": "returns", "exchange": "returns", "refund": "returns",
    "policy": "policy", "expire": "expire", "fee": "fee",
}
_AXES = ("timeoff", "giftcard", "returns", "policy", "expire", "fee")


def _sem_embed(text: str) -> list[float]:
    words = text.lower().replace("?", " ").split()
    axes = {_SYN[w] for w in words if w in _SYN}
    v = [1.0 if a in axes else 0.0 for a in _AXES]
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v] if norm else v


class _EchoSynth:
    """Understanding echoes the chunk texts, so the digest carries the source's topic
    words for the axis embedder (a controllable stand-in for a real digest)."""

    def synthesize(self, query: str, chunks: list[Chunk]) -> Synthesis:
        claims = [c.text for c in chunks]
        return Synthesis(
            understanding={"summary": " ".join(claims), "claims": claims},
            used=list(range(len(chunks))),
        )


def _kb() -> InMemoryRetriever:
    retriever = InMemoryRetriever(top_k=1)
    retriever.add("hr:leave", "leave policy timeoff vacation days")
    retriever.add("shop:giftcard", "gift card expire fee")
    retriever.add("shop:returns", "returns exchange policy refund")
    return retriever


def _cache() -> SemanticCache:
    return SemanticCache(
        _kb(), _EchoSynth(), embedder=FunctionEmbedder(_sem_embed), hit_threshold=0.55
    )


def test_paraphrase_hits_the_right_unit() -> None:
    cache = _cache()
    leave = cache.get("leave policy")
    cache.get("gift card")
    # 'vacation days' shares NO surface words with 'leave policy' but the same topic axis.
    result = cache.get("how many vacation days")
    assert result.cache_hit is True
    assert result.unit_id == leave.unit_id


def test_surface_form_collision_is_not_a_false_hit() -> None:
    cache = _cache()
    leave = cache.get("leave policy")
    returns = cache.get("returns exchange policy")
    # 'exchange policy' shares the word 'policy' with leave, but its TOPIC is returns.
    result = cache.get("exchange policy")
    assert result.unit_id == returns.unit_id
    assert result.unit_id != leave.unit_id


def test_seed_query_does_not_drift_on_rematerialize() -> None:
    cache = _cache()
    built = cache.get("leave policy")
    original_seed = cache._units[built.unit_id].query

    cache.source_changed("hr:leave", text="leave policy timeoff updated")  # dirties it
    cache.get("vacation days entitlement")  # re-materializes the SAME unit, different query
    assert cache._units[built.unit_id].query == original_seed  # seed unchanged (no drift)


def test_legacy_unit_backfills_on_load(tmp_path) -> None:  # type: ignore[no-untyped-def]
    db = str(tmp_path / "c.db")

    store1 = SQLiteCognitionStore(db)
    c1 = SemanticCache(
        _kb(), _EchoSynth(), embedder=FunctionEmbedder(_sem_embed), store=store1, hit_threshold=0.55
    )
    built = c1.get("leave policy")
    unit = c1._units[built.unit_id]
    unit.understanding_embedding = ()  # simulate a pre-v0.3 record (no embeddings)
    unit.claim_embeddings = ()
    store1.put(unit)
    store1.close()

    store2 = SQLiteCognitionStore(db)
    c2 = SemanticCache(
        _kb(), _EchoSynth(), embedder=FunctionEmbedder(_sem_embed), store=store2, hit_threshold=0.55
    )
    reloaded = c2._units[built.unit_id]
    assert reloaded.understanding_embedding != ()  # backfilled on load
    assert c2.get("how many vacation days").unit_id == built.unit_id  # and matches by meaning
    store2.close()
