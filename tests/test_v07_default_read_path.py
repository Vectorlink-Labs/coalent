"""v0.7 BREAKING — the default read path RESOLVES (user ruling 2026-09-15).

``read_path=None`` (the new constructor default) resolves at construction time:

* a SEMANTIC embedder (anything but the lexical ``HashingEmbedder`` — the exact
  classification the pool guard already enforces) -> ``"pool"``, the measured path
  behind every published v0.6/v0.7 number;
* the ``HashingEmbedder`` (the keyless zero-config fallback) -> ``"unit"``, LOUDLY:
  a ``UserWarning`` names the resolution rule and both remedies (set
  ``OPENAI_API_KEY`` / pass ``embedder=``);
* an EXPLICIT ``read_path`` always wins and behaves exactly as pre-0.7 — including
  the pool-requires-semantic-embedder constructor error, which stays pinned both in
  ``test_v06_pool_read_path.test_pool_requires_semantic_embedder`` and here.

The escape-hatch byte-identity (explicit ``read_path="unit"`` == the pre-flip unit
path, and the Hashing-resolved default collapsing onto it) is pinned in
``test_v06_pool_read_path.test_unit_mode_bit_identical``.
"""
from __future__ import annotations

import inspect
import warnings

import pytest

from coalent import FunctionEmbedder
from coalent.semantic import (
    Chunk,
    HashingEmbedder,
    InMemoryRetriever,
    SemanticCache,
    Synthesis,
)

_AXES = ("alpha", "beta", "gamma", "value", "one", "two", "other")


def _embed(text: str) -> list[float]:
    words = set(text.lower().split())
    v = [1.0 if a in words else 0.0 for a in _AXES]
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v] if n else v


class _EchoSynth:
    def synthesize(self, query: str, chunks: list[Chunk]) -> Synthesis:
        text = " ".join(c.text for c in chunks)
        return Synthesis(understanding={"summary": text, "claims": [text]},
                         used=list(range(len(chunks))))


def _world() -> InMemoryRetriever:
    ret = InMemoryRetriever()
    ret.add("src:a", "alpha value one")
    ret.add("src:b", "beta value two")
    return ret


# ------------------------------------------------------------------- the signature pin

def test_read_path_defaults_to_none_sentinel() -> None:
    assert inspect.signature(SemanticCache.__init__).parameters["read_path"].default is None


# ------------------------------------------------- pin 1: semantic embedder -> pool

def test_default_resolves_to_pool_under_semantic_embedder() -> None:
    # The bare default under a semantic embedder IS the pool path — and it keeps the
    # pool path's own loudness ladder (the pool_header attribution warning fires,
    # exactly as an explicit read_path="pool" construction without a header warns).
    with pytest.warns(UserWarning, match="pool_header"):
        cache = SemanticCache(_world(), _EchoSynth(),             # type: ignore[arg-type]
                              embedder=FunctionEmbedder(_embed),
                              hit_threshold=0.99, coverage_floor=0.0)
    assert cache._read_path == "pool"
    assert cache.stats()["read_path"] == "pool"
    assert cache._serve_budget == 1000            # the pool-path serve_budget default


def test_resolved_pool_is_bit_identical_to_explicit_pool() -> None:
    def mk(**kw: object) -> SemanticCache:
        return SemanticCache(_world(), _EchoSynth(),              # type: ignore[arg-type]
                             embedder=FunctionEmbedder(_embed),
                             hit_threshold=0.99, coverage_floor=0.0,
                             clock=lambda: 1000.0,
                             pool_header=lambda u: "[hdr]", **kw)  # type: ignore[arg-type]

    resolved = mk()                     # sentinel -> "pool"
    explicit = mk(read_path="pool")
    for q in ("alpha value one", "beta value two", "alpha value one"):
        assert resolved.get(q) == explicit.get(q)
    resolved.source_changed("src:a", text="alpha value one revised")
    explicit.source_changed("src:a", text="alpha value one revised")
    assert resolved.get("alpha value one") == explicit.get("alpha value one")
    assert resolved.stats() == explicit.stats()


# ------------------------------------- pin 2: HashingEmbedder -> unit, LOUDLY

def test_default_falls_back_to_unit_under_hashing_and_warns() -> None:
    with pytest.warns(UserWarning) as rec:
        cache = SemanticCache(_world(), _EchoSynth(),             # type: ignore[arg-type]
                              embedder=HashingEmbedder())
    assert cache._read_path == "unit"
    assert cache._serve_budget == 600             # the unit-path serve_budget default
    msgs = [str(w.message) for w in rec if "read_path was not set" in str(w.message)]
    assert len(msgs) == 1, "the fallback warning must fire exactly once"
    # The warning names the resolution rule and BOTH remedies, plus the silencer.
    assert "pool" in msgs[0] and "semantic embedder" in msgs[0]
    assert "OPENAI_API_KEY" in msgs[0]
    assert "embedder=" in msgs[0]
    assert "read_path='unit'" in msgs[0]


# --------------------------- pin 3: explicit read_path always wins, silently

def test_explicit_read_path_wins_and_never_warns() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        unit_hash = SemanticCache(_world(), _EchoSynth(),         # type: ignore[arg-type]
                                  embedder=HashingEmbedder(), read_path="unit")
        unit_sem = SemanticCache(_world(), _EchoSynth(),          # type: ignore[arg-type]
                                 embedder=FunctionEmbedder(_embed), read_path="unit")
        pool_sem = SemanticCache(_world(), _EchoSynth(),          # type: ignore[arg-type]
                                 embedder=FunctionEmbedder(_embed), read_path="pool",
                                 pool_header=lambda u: "[hdr]")
    assert unit_hash._read_path == "unit"
    assert unit_sem._read_path == "unit"
    assert pool_sem._read_path == "pool"


# --------------- pin 4: the pool-requires-semantic-embedder guard is unchanged

def test_explicit_pool_under_hashing_still_raises() -> None:
    with pytest.raises(ValueError, match="semantic embedder"):
        SemanticCache(_world(), _EchoSynth(),                     # type: ignore[arg-type]
                      embedder=HashingEmbedder(), read_path="pool")
    # And the conftest default (HashingEmbedder) behaves the same on explicit pool.
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        SemanticCache(_world(), _EchoSynth(), read_path="pool")   # type: ignore[arg-type]
