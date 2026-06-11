"""Acceptance test — light, lazy cross-unit relationships.

Proves units link by shared entity and by shared source, that a read folds in
related units ranked by query relevance, and that isolated units surface none.
NOT a graph engine — just enough to enable cross-unit reuse.
"""
from __future__ import annotations

from coalent.semantic import Chunk, InMemoryRetriever, SemanticCache, Synthesis


class FakeSynth:
    """Synthesizer that emits controlled entities per query (cites all chunks)."""

    def __init__(self, entities_by_query: dict[str, list[str]]) -> None:
        self._map = entities_by_query

    def synthesize(self, query: str, chunks: list[Chunk]) -> Synthesis:
        entities = self._map.get(query, [])
        return Synthesis(
            understanding={"summary": query, "entities": entities},
            used=list(range(len(chunks))),
        )


def test_units_link_by_shared_entity() -> None:
    retriever = InMemoryRetriever(top_k=1)  # one doc each -> distinct provenance
    retriever.add("doc:leave", "annual leave policy twentyone vacation days")
    retriever.add("doc:parental", "parental newborn caregiver entitlement weeks")
    cache = SemanticCache(
        retriever,
        FakeSynth(
            {
                "leave policy": ["leave", "vacation"],
                "parental newborn entitlement": ["leave", "parental"],
            }
        ),
    )

    a = cache.get("leave policy")
    b = cache.get("parental newborn entitlement")
    assert a.unit_id != b.unit_id  # distinct intents

    result = cache.get("tell me about leave")
    assert result.related, "expected a related unit via the shared 'leave' entity"
    other = b.unit_id if result.unit_id == a.unit_id else a.unit_id
    assert other in {r.unit_id for r in result.related}
    assert any(r.relation == "shared_entity" for r in result.related)


def test_units_link_by_shared_source() -> None:
    retriever = InMemoryRetriever(top_k=1)
    retriever.add("confluence:handbook", "remote work and wifi access policy in the handbook")
    cache = SemanticCache(retriever, FakeSynth({}))  # no entities -> only source links

    a = cache.get("remote work policy")
    b = cache.get("wifi access policy")
    assert a.unit_id != b.unit_id

    result = cache.get("remote work")
    assert any(r.relation == "shared_source" for r in result.related)
    other = b.unit_id if result.unit_id == a.unit_id else a.unit_id
    assert other in {r.unit_id for r in result.related}


def test_isolated_unit_has_no_related() -> None:
    retriever = InMemoryRetriever(top_k=1)
    retriever.add("doc:only", "a standalone document about quarterly travel reimbursement")
    cache = SemanticCache(retriever, FakeSynth({}))
    result = cache.get("travel reimbursement")
    assert result.related == []


def test_related_respects_limit_and_excludes_self() -> None:
    retriever = InMemoryRetriever(top_k=1)
    for i in range(4):
        retriever.add(f"doc:{i}", f"benefit topic number {i} about leave coverage")
    cache = SemanticCache(retriever, FakeSynth({f"q{i}": ["leave"] for i in range(4)}))
    for i in range(4):
        cache.get(f"q{i}")

    result = cache.get("q0", related=2)
    assert len(result.related) <= 2
    assert all(r.unit_id != result.unit_id for r in result.related)
