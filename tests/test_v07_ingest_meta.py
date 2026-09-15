"""v0.7 — ingest metadata (V0.7-SPEC.md, locked 2026-08-11), regression-pinned.

The chain, each link pinned here: ``Chunk.meta`` (additive, default None — zero impact
on existing callers) -> ``unit.source_meta`` captured at build (DOMINANT artifact,
first chunk with meta wins) -> serde round-trip (written only when set, so pre-v0.7
JSON stays byte-identical) -> the metadata rung of the default pool-header ladder
(rung ORDER is pinned in test_v06_pool_read_path.test_pool_payload_always_attributed;
the end-to-end serve is pinned here). Offline-deterministic throughout.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from coalent import FunctionEmbedder
from coalent.semantic import Chunk, InMemoryRetriever, SemanticCache, Synthesis
from coalent.semantic.serde import cognition_from_dict, cognition_to_dict

_AXES = ("alpha", "beta", "gamma", "value", "one", "two")


def _embed(text: str) -> list[float]:
    words = set(text.lower().split())
    v = [1.0 if a in words else 0.0 for a in _AXES]
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v] if n else v


class _Synth:
    """Echoes chunk text into a single claim — enough to drive real builds offline."""

    def synthesize(self, query: str, chunks: list[Chunk]) -> Synthesis:
        text = " ".join(c.text for c in chunks)
        return Synthesis(understanding={"summary": text, "claims": [text]},
                         used=list(range(len(chunks))))


class _MutableRetriever:
    """Word-overlap retriever whose docs AND meta can be edited — drives rebuilds."""

    def __init__(self) -> None:
        self.docs: dict[str, Chunk] = {}

    def set(self, artifact_id: str, text: str, meta: dict[str, str] | None = None) -> None:
        self.docs[artifact_id] = Chunk(artifact_id=artifact_id, text=text, meta=meta)

    def retrieve(self, query: str, *, namespace: str | None = None) -> list[Chunk]:
        qs = set(query.lower().split())
        return [c for c in self.docs.values() if qs & set(c.text.lower().split())]


def _pool(retriever: object, **kw: object) -> SemanticCache:
    kw.setdefault("coverage_floor", 0.0)
    return SemanticCache(retriever, _Synth(), embedder=FunctionEmbedder(_embed),  # type: ignore[arg-type]
                         read_path="pool", **kw)  # type: ignore[arg-type]


# ------------------------------------------------------------------- the Chunk surface

def test_chunk_meta_additive_default_none() -> None:
    # The pre-v0.7 positional shape (4 fields) constructs unchanged; meta defaults None.
    c = Chunk("art:1", "text", "v2", "hash")
    assert c.meta is None
    assert c == Chunk(artifact_id="art:1", text="text", version="v2", content_hash="hash")
    m = Chunk("art:1", "text", meta={"title": "T", "extra": "kept"})
    assert m.meta == {"title": "T", "extra": "kept"}
    with pytest.raises(dataclasses.FrozenInstanceError):   # frozen stays frozen
        m.meta = {}  # type: ignore[misc]


# ------------------------------------------------------- build-time capture (the rule)

def test_dominant_meta_rule() -> None:
    # Spec: source_meta = the DOMINANT artifact's chunk meta, first chunk with meta wins.
    dom = SemanticCache._dominant_meta
    a1 = Chunk("art:A", "a1")                              # dominant artifact, meta-less
    a2 = Chunk("art:A", "a2", meta={"title": "A2"})        # first WITH meta -> wins
    a3 = Chunk("art:A", "a3", meta={"title": "A3"})
    b1 = Chunk("art:B", "b1", meta={"title": "B"})
    assert dom([a1, a2, a3, b1]) == {"title": "A2"}        # most chunks -> art:A
    # A count tie breaks to the first-seen artifact (the split path's dominance rule).
    assert dom([b1, a2]) == {"title": "B"}
    # A meta-less dominant artifact yields EMPTY meta even when a minority chunk has
    # meta — the header must never attribute a unit to a source that did not dominate it.
    assert dom([a1, Chunk("art:A", "a1b"), b1]) == {}
    assert dom([]) == {}


def test_source_meta_captured_at_build_and_recomputed_on_rebuild() -> None:
    ret = _MutableRetriever()
    meta = {"title": "Alpha Report", "source": "feed:a", "date": "2026-08-01", "x": "extra"}
    ret.set("src:a", "alpha value one", meta=meta)
    cache = _pool(ret, serve_gate=0.4, pool_header=lambda u: "[h]")
    cache.get("alpha value one")
    unit = next(iter(cache._units.values()))
    assert unit.source_meta == meta                        # extras preserved, unused
    # A rebuild recomputes meta from the CURRENT evidence: gone at the source -> gone
    # on the unit (header material never outlives the sources it names).
    ret.set("src:a", "alpha value two", meta=None)
    cache.source_changed("src:a")
    cache.get("alpha value two")
    assert unit.source_meta == {}


# ------------------------------------------------------------------- serde round-trip

def test_source_meta_serde_round_trip() -> None:
    ret = _MutableRetriever()
    meta = {"title": "Alpha Report", "source": "feed:a", "date": "2026-08-01"}
    ret.set("src:a", "alpha value one", meta=meta)
    cache = _pool(ret, serve_gate=0.4, pool_header=lambda u: "[h]")
    cache.get("alpha value one")
    unit = next(iter(cache._units.values()))
    d = cognition_to_dict(unit)
    assert d["source_meta"] == meta
    assert d["evidence"][0]["meta"] == meta                # chunk meta persists too
    back = cognition_from_dict(json.loads(json.dumps(d)))  # through REAL JSON
    assert back.source_meta == meta
    assert back.evidence[0].meta == meta
    # Written ONLY when set: a meta-less unit emits byte-identical pre-v0.7 JSON keys,
    # and pre-v0.7 payloads (no source_meta key) load to empty meta.
    unit.source_meta = {}
    unit.evidence = (Chunk("src:a", "alpha value one"),)
    d2 = cognition_to_dict(unit)
    assert "source_meta" not in d2
    assert "meta" not in d2["evidence"][0]
    assert cognition_from_dict(d2).source_meta == {}
    assert cognition_from_dict(d2).evidence[0].meta is None


# ------------------------------------------------------------- end-to-end pool serving

def test_meta_header_serves_end_to_end() -> None:
    # InMemoryRetriever.add(meta=...) -> build -> the DEFAULT ladder's metadata rung
    # heads the served group; a meta-less source in the same payload keeps its query rung.
    ret = InMemoryRetriever()
    ret.add("src:a", "alpha value one",
            meta={"title": "Alpha Report", "source": "feed:a", "date": "2026-08-01"})
    ret.add("src:b", "beta value two")                     # no meta -> "## " rung
    cache = _pool(ret, serve_gate=0.4)                     # NO pool_header: the ladder
    cache.get("alpha value one")                           # builds both artifacts
    r = cache.get("value one")
    heads = {g.splitlines()[0] for g in r.context["pool"].split("\n\n")}
    assert heads == {"[Alpha Report | feed:a | 2026-08-01]", "## alpha value one"}
