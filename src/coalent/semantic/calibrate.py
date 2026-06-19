"""Threshold calibration for the semantic cache.

Cosine distributions are embedder-specific, so the right ``hit_threshold`` /
``coverage_floor`` depend on YOUR embedder. ``SemanticCache`` already picks sensible
defaults for the shipped embedders (``default_thresholds_for``); use these helpers to
tune precisely — especially for a custom/local embedder, or OpenAI
``text-embedding-3-large`` whose distribution differs from ``-3-small``.

    from coalent import calibrate_thresholds, SemanticCache

    t = calibrate_thresholds(
        embedder,
        positives=[("how many vacation days", "Leave policy: 21 days annual leave...")],
        negatives=[("what is the exchange policy", "Leave policy: 21 days annual leave...")],
    )
    cache = SemanticCache(retriever, synth, embedder=embedder, **t)
"""
from __future__ import annotations

from .embedding import Embedder, cosine, embed_texts


def _pair_cosines(embedder: Embedder, pairs: list[tuple[str, str]]) -> list[float]:
    """Embed both sides of each pair (batched) and return their cosine similarities."""
    texts = [side for pair in pairs for side in pair]
    vecs = embed_texts(embedder, texts)
    return [cosine(vecs[2 * i], vecs[2 * i + 1]) for i in range(len(pairs))]


def calibrate_thresholds(
    embedder: Embedder,
    positives: list[tuple[str, str]],
    negatives: list[tuple[str, str]],
) -> dict[str, float]:
    """Tune ``(hit_threshold, coverage_floor)`` from labeled ``(query, understanding)``
    pairs. ``positives`` SHOULD match (a query and an understanding that genuinely
    answers it); ``negatives`` should NOT (e.g. "exchange policy" vs a leave-policy
    understanding). Sets ``hit_threshold`` midway between the worst positive and the
    best negative (separating them), and ``coverage_floor`` a notch below."""
    pos = sorted(_pair_cosines(embedder, positives)) if positives else [0.4]
    neg = sorted(_pair_cosines(embedder, negatives)) if negatives else [0.2]
    worst_pos, best_neg = pos[0], neg[-1]
    hit = round((worst_pos + best_neg) / 2.0, 3)
    return {"hit_threshold": hit, "coverage_floor": round(max(0.0, hit - 0.05), 3)}


def suggest_thresholds(embedder: Embedder, texts: list[str]) -> dict[str, float]:
    """Labels-free starting point for ANY embedder: estimate the 'unrelated' cosine
    baseline from a DIVERSE set of ``texts`` (assumed mostly unrelated to one another)
    and set the bar a margin above the 95th percentile of those baseline cosines.
    Approximate — prefer ``calibrate_thresholds`` when you have labeled pairs."""
    vecs = embed_texts(embedder, texts)
    sims = sorted(
        cosine(vecs[i], vecs[j])
        for i in range(len(vecs))
        for j in range(i + 1, len(vecs))
    )
    if not sims:
        return {"hit_threshold": 0.45, "coverage_floor": 0.4}
    p95 = sims[min(len(sims) - 1, int(0.95 * len(sims)))]
    hit = round(min(0.9, p95 + 0.08), 3)
    return {"hit_threshold": hit, "coverage_floor": round(max(0.0, hit - 0.05), 3)}
