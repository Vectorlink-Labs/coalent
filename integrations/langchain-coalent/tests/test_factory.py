"""LG5 — knob passthrough: every Coalent knob reaches the SemanticCache constructor
through create_coalent_cache, plus the factory's guard rails and recommended defaults."""
from __future__ import annotations

import json

import pytest
from conftest import BagOfWordsEmbeddings, FakeVectorStore, ScriptedChatModel

from coalent import Chunk, InMemoryRetriever, StubSynthesizer, Synthesis
from langchain_coalent import create_coalent_cache


def _vs(embeddings: BagOfWordsEmbeddings) -> FakeVectorStore:
    vs = FakeVectorStore(embeddings)
    vs.add_texts(["The leave allowance is 21 days."], metadatas=[{"source": "policy:leave"}])
    return vs


def test_lg5_knobs_reach_the_constructor(embeddings: BagOfWordsEmbeddings) -> None:
    cache = create_coalent_cache(
        _vs(embeddings),
        embeddings=embeddings,
        key_floor=0.91,
        hit_margin=0.07,
        serve_budget=777,
        recall_limit=9,
        residual_spans=True,
        query_keys=True,
        serve_gate=0.42,
    )
    # No public accessors exist for knob values, so the passthrough proof reads the
    # constructor-set fields directly (documented private peek, tests only).
    assert cache._key_floor == 0.91
    assert cache._hit_margin == 0.07
    assert cache._serve_budget == 777
    assert cache._recall_limit == 9
    assert cache._residual_spans is True
    assert cache._query_keys is True
    assert cache._serve_gate == 0.42


def test_lg5_unknown_knob_fails_loud(embeddings: BagOfWordsEmbeddings) -> None:
    with pytest.raises(TypeError, match="no_such_knob"):
        create_coalent_cache(_vs(embeddings), embeddings=embeddings, no_such_knob=1)


def test_lg5_bad_knob_value_fails_loud(embeddings: BagOfWordsEmbeddings) -> None:
    with pytest.raises(ValueError, match="preset"):
        create_coalent_cache(_vs(embeddings), embeddings=embeddings, preset="not-a-preset")


def test_default_read_path_pool_with_semantic_embeddings(
    embeddings: BagOfWordsEmbeddings,
) -> None:
    cache = create_coalent_cache(_vs(embeddings), embeddings=embeddings)
    assert cache._read_path == "pool"
    assert cache._pool_header is not None          # attribution header default applied


def test_default_read_path_unit_without_embeddings(
    embeddings: BagOfWordsEmbeddings, no_openai_env: None
) -> None:
    with pytest.warns(UserWarning):                # coalent's hashing-fallback warning
        cache = create_coalent_cache(_vs(embeddings))
    assert cache._read_path == "unit"              # pool requires a semantic embedder


def test_explicit_read_path_wins(embeddings: BagOfWordsEmbeddings) -> None:
    cache = create_coalent_cache(_vs(embeddings), embeddings=embeddings, read_path="unit")
    assert cache._read_path == "unit"


def test_adapter_knob_k_reaches_the_wrapper(embeddings: BagOfWordsEmbeddings) -> None:
    vs = FakeVectorStore(embeddings)
    vs.add_texts(["alpha one", "beta two", "gamma three"])
    cache = create_coalent_cache(vs, embeddings=embeddings, k=1)
    assert len(cache._retriever.retrieve("alpha one")) == 1


def test_byo_coalent_retriever_used_as_is(embeddings: BagOfWordsEmbeddings) -> None:
    byo = InMemoryRetriever()
    byo.add("src:1", "some text")
    cache = create_coalent_cache(byo, embeddings=embeddings)
    assert cache._retriever is byo


def test_byo_retriever_rejects_adapter_knobs(embeddings: BagOfWordsEmbeddings) -> None:
    with pytest.raises(TypeError, match="already a Coalent Retriever"):
        create_coalent_cache(InMemoryRetriever(), embeddings=embeddings, k=2)


def test_rejects_unknown_substrate(embeddings: BagOfWordsEmbeddings) -> None:
    with pytest.raises(TypeError, match="VectorStore"):
        create_coalent_cache(object(), embeddings=embeddings)


def test_embeddings_and_embedder_conflict(embeddings: BagOfWordsEmbeddings) -> None:
    from langchain_coalent import LangChainEmbedder

    with pytest.raises(TypeError, match="not both"):
        create_coalent_cache(
            _vs(embeddings), embeddings=embeddings, embedder=LangChainEmbedder(embeddings)
        )


def test_llm_and_synthesizer_conflict(embeddings: BagOfWordsEmbeddings) -> None:
    chat = ScriptedChatModel(script=lambda p: "{}")
    with pytest.raises(TypeError, match="not both"):
        create_coalent_cache(
            _vs(embeddings), llm=chat, embeddings=embeddings, synthesizer=StubSynthesizer()
        )


def test_synth_knobs_require_llm(embeddings: BagOfWordsEmbeddings) -> None:
    with pytest.raises(TypeError, match="llm"):
        create_coalent_cache(_vs(embeddings), embeddings=embeddings, extract=False)


def test_synth_knobs_shape_the_synthesizer(embeddings: BagOfWordsEmbeddings) -> None:
    seen: list[str] = []

    def script(prompt: str) -> str:
        seen.append(prompt)
        return json.dumps({"summary": "s", "claims": [], "entities": [], "facts": {}, "used": [0]})

    chat = ScriptedChatModel(script=script)
    cache = create_coalent_cache(
        _vs(embeddings), llm=chat, embeddings=embeddings,
        instruction="CUSTOM-INSTRUCTION-MARKER", extract=False,
    )
    cache.get("what is the leave allowance?")
    assert seen and "CUSTOM-INSTRUCTION-MARKER" in seen[0]


def test_custom_synthesizer_passthrough(embeddings: BagOfWordsEmbeddings) -> None:
    class MySynth:
        def synthesize(self, query: str, chunks: list[Chunk]) -> Synthesis:  # pragma: no cover
            raise NotImplementedError

    synth = MySynth()
    cache = create_coalent_cache(_vs(embeddings), embeddings=embeddings, synthesizer=synth)
    assert cache._synth is synth
