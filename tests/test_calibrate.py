"""Phase C — threshold calibration + embedder-aware defaults.

Proves:
  1. calibrate_thresholds separates labeled positives from negatives.
  2. suggest_thresholds returns a sane labels-free starting point.
  3. default_thresholds_for gives embedder-appropriate defaults (the fix for
     "0.6 is too high for OpenAI" — OpenAI gets a lower bar than HashingEmbedder).
  4. SemanticCache derives its threshold from the embedder when unset, and an
     explicit override wins.
"""
from __future__ import annotations

from coalent import (
    FunctionEmbedder,
    HashingEmbedder,
    InMemoryRetriever,
    OpenAIEmbedder,
    SemanticCache,
    StubSynthesizer,
    calibrate_thresholds,
    default_thresholds_for,
    suggest_thresholds,
)


class _FakeClient:
    """Lets us construct OpenAIEmbedder without the network or the openai package."""

    def __init__(self) -> None:
        self.embeddings = self


def test_calibrate_separates_positives_from_negatives() -> None:
    vecs = {
        "qa": [1.0, 0.0], "ua": [1.0, 0.05],   # ~1.0 cosine (positive)
        "qb": [0.0, 1.0], "ub": [0.05, 1.0],   # ~1.0 cosine (positive)
        "nq": [1.0, 0.0], "nu": [0.0, 1.0],    # 0.0 cosine (negative)
    }
    emb = FunctionEmbedder(lambda t: vecs[t])
    res = calibrate_thresholds(
        emb,
        positives=[("qa", "ua"), ("qb", "ub")],
        negatives=[("nq", "nu")],
    )
    hit = res["hit_threshold"]
    assert 0.0 < hit < 0.999  # strictly separates best-negative (0) from worst-positive (~1)
    assert res["coverage_floor"] < hit  # coverage bar sits a notch below the hit bar


def test_suggest_thresholds_from_unrelated_baseline() -> None:
    vecs = {"a": [1.0, 0.0, 0.0], "b": [0.0, 1.0, 0.0], "c": [0.0, 0.0, 1.0]}  # orthogonal
    emb = FunctionEmbedder(lambda t: vecs[t])
    res = suggest_thresholds(emb, ["a", "b", "c"])
    # unrelated baseline is 0 -> bar lands just above it (margin), with coverage below.
    assert res["hit_threshold"] == 0.08
    assert res["coverage_floor"] == 0.03


def test_default_thresholds_are_embedder_aware() -> None:
    # OpenAI cosines are compressed -> a LOWER bar than the lexical HashingEmbedder.
    assert default_thresholds_for(OpenAIEmbedder(client=_FakeClient())) == (0.33, 0.28)
    assert default_thresholds_for(HashingEmbedder()) == (0.6, 0.5)
    assert default_thresholds_for(FunctionEmbedder(lambda t: [1.0])) == (0.45, 0.4)


def test_cache_derives_threshold_from_embedder() -> None:
    cache = SemanticCache(InMemoryRetriever(), StubSynthesizer(), embedder=HashingEmbedder())
    assert cache._threshold == 0.6  # derived from HashingEmbedder
    assert cache._coverage_floor == 0.5

    override = SemanticCache(
        InMemoryRetriever(), StubSynthesizer(), embedder=HashingEmbedder(), hit_threshold=0.1
    )
    assert override._threshold == 0.1  # explicit override wins
