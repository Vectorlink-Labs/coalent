"""v0.4 — cross-unit claim recall, accuracy tier (REAL OpenAI embeddings).

The validation of record: the logic tier (test_v04_recall.py) pins the algorithm with a
controllable embedder; here we prove the mechanism survives on genuine semantic embeddings,
where claim cosine reflects MEANING rather than keyword overlap. Skipped offline.

Scenario is a clean 2-hop: the answer needs hop-1 (Alice -> France) from one paragraph and
hop-2 (France -> Paris) from another. A single best-match unit can hold only one hop;
cross-unit recall must recover the other.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

pytestmark = pytest.mark.openai

# Both the key AND the SDK: with a key exported but the openai package absent
# (plain `pip install -e ".[dev]"`), these must SKIP, not crash on import.
_needs_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY") or importlib.util.find_spec("openai") is None,
    reason="v0.4 accuracy tier needs OPENAI_API_KEY and the openai SDK installed",
)

DOCS = {
    "alice": "Alice lives in France.",
    "france": "The capital of France is Paris.",
    "bob": "Bob lives in Germany.",
    "japan": "The capital of Japan is Tokyo.",
}
QUESTION = "What is the capital of the country Alice lives in?"


class _AtomSynth:
    def synthesize(self, query, chunks):  # type: ignore[no-untyped-def]
        from coalent.semantic import Synthesis
        text = " ".join(c.text for c in chunks)
        claims = [s.strip().rstrip(".") for s in text.split(". ") if s.strip()]
        return Synthesis(understanding={"summary": text, "claims": claims or [text]}, used=[0])


def _warm(docs, **kw):  # type: ignore[no-untyped-def]
    """A warm cache with one unit per paragraph, on real OpenAI embeddings."""
    from coalent.semantic import Chunk, FunctionRetriever, OpenAIEmbedder, SemanticCache
    holder: dict[str, object] = {}
    cache = SemanticCache(
        FunctionRetriever(lambda q, ns: [holder["chunk"]]), _AtomSynth(),
        embedder=OpenAIEmbedder(), hit_threshold=0.0, enable_coverage_escalation=False, **kw,
    )
    query_threshold = cache._threshold
    cache._threshold = 2.0
    for doc_id, text in docs.items():
        holder["chunk"] = Chunk(artifact_id=doc_id, text=text)
        cache.get(text)
    cache._threshold = query_threshold
    doc_of_unit = {u.id: u.evidence[0].artifact_id for u in cache._units.values() if u.evidence}
    return cache, doc_of_unit


@_needs_key
def test_cross_unit_recovers_the_bridge_fact() -> None:
    cache, doc_of_unit = _warm(DOCS, cross_unit_recall=True, recall_threshold=0.9)
    result = cache.get(QUESTION)

    assert result.recalled, "cross-unit recall should fire on this under-covered multi-hop query"
    recalled_docs = {doc_of_unit.get(r.unit_id) for r in result.recalled}
    assert "france" in recalled_docs                      # the hop-2 fact is recovered...
    assert recalled_docs - {doc_of_unit.get(result.unit_id)}  # ...from a unit beyond the best match


@_needs_key
def test_single_unit_cannot_hold_both_hops() -> None:
    cache, _doc_of_unit = _warm(DOCS)                      # flag off — pure v0.3 behavior
    result = cache.get(QUESTION)

    assert result.recalled == []
    u = result.context.get("understanding", {})
    served = " ".join([str(u.get("summary", "")), *[str(c) for c in (u.get("claims") or [])]])
    # one paragraph holds the 'Alice' hop OR the 'Paris' hop — never both. That is the gap.
    assert not ("Alice" in served and "Paris" in served)
