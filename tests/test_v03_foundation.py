"""Phase A (v0.3 foundation) — data-model, serde, and batch-embedding tests.

Proves:
  1. The new Cognition fields round-trip through serde, and v0.2 JSON (which lacks
     them) loads with empty defaults -> needs_backfill (the migration signal).
  2. touch(query) records hitting queries, deduped and bounded.
  3. embed_texts batches via embed_many when available, else loops embed.
"""
from __future__ import annotations

from coalent.domain.models import ProvenanceManifest
from coalent.semantic import FunctionEmbedder, HashingEmbedder
from coalent.semantic.embedding import embed_texts
from coalent.semantic.serde import cognition_from_dict, cognition_to_dict
from coalent.semantic.unit import Cognition


def _unit(**over: object) -> Cognition:
    base: dict[str, object] = dict(
        id="cog:1",
        namespace="",
        query="leave policy",
        query_embedding=(0.1, 0.2),
        understanding={"summary": "21 days leave", "claims": ["21 days annual leave"]},
        evidence=(),
        provenance=ProvenanceManifest("synth@1", "semantic@2"),
    )
    base.update(over)
    return Cognition(**base)  # type: ignore[arg-type]


def test_new_fields_round_trip_through_serde() -> None:
    unit = _unit(
        understanding_embedding=(0.5, 0.5),
        claim_embeddings=((0.1, 0.9), (0.9, 0.1)),
        hit_queries=("how many vacation days", "pto allowance"),
    )
    restored = cognition_from_dict(cognition_to_dict(unit))
    assert restored.understanding_embedding == (0.5, 0.5)
    assert restored.claim_embeddings == ((0.1, 0.9), (0.9, 0.1))
    assert restored.hit_queries == ("how many vacation days", "pto allowance")
    assert restored.needs_backfill is False


def test_legacy_v02_json_loads_with_defaults_and_needs_backfill() -> None:
    # A v0.2 record predates the new keys entirely.
    legacy = {
        "id": "cog:old",
        "namespace": "",
        "query": "leave policy",
        "query_embedding": [0.1, 0.2],
        "understanding": {"summary": "21 days"},
        "evidence": [],
        "provenance": {
            "model_version": "synth@1",
            "prompt_version": "semantic@2",
            "source_spans": [],
            "observed_edges": [],
        },
        "status": "fresh",
    }
    unit = cognition_from_dict(legacy)
    assert unit.understanding_embedding == ()
    assert unit.claim_embeddings == ()
    assert unit.hit_queries == ()
    assert unit.needs_backfill is True  # -> will be backfilled on load


def test_touch_records_bounded_deduped_hit_queries() -> None:
    unit = _unit()
    unit.touch("q1", max_queries=3)
    unit.touch("q1", max_queries=3)  # dedup: same query not re-added
    unit.touch("q2", max_queries=3)
    unit.touch("q3", max_queries=3)
    unit.touch("q4", max_queries=3)  # over the cap -> evicts oldest (q1)
    assert unit.hit_queries == ("q2", "q3", "q4")
    assert unit.hits == 5  # counter increments on every call
    unit.touch()  # no query -> just counts, no hit_queries change
    assert unit.hits == 6
    assert unit.hit_queries == ("q2", "q3", "q4")


def test_embed_texts_loops_for_single_method_embedder() -> None:
    emb = FunctionEmbedder(lambda t: [float(len(t))])
    assert embed_texts(emb, ["a", "bb", "ccc"]) == [[1.0], [2.0], [3.0]]
    assert embed_texts(emb, []) == []


def test_embed_texts_uses_batch_when_available() -> None:
    calls = {"batch": 0, "single": 0}

    def single(t: str) -> list[float]:
        calls["single"] += 1
        return [float(len(t))]

    def batch(texts: list[str]) -> list[list[float]]:
        calls["batch"] += 1
        return [[float(len(t)) * 10] for t in texts]

    emb = FunctionEmbedder(single, batch=batch)
    out = embed_texts(emb, ["a", "bb"])
    assert out == [[10.0], [20.0]]
    assert calls["batch"] == 1  # one batched call...
    assert calls["single"] == 0  # ...not per-string


def test_hashing_embedder_loops_via_embed_texts() -> None:
    emb = HashingEmbedder(dim=16)
    out = embed_texts(emb, ["leave policy", "gift card"])
    assert len(out) == 2
    assert all(len(v) == 16 for v in out)
