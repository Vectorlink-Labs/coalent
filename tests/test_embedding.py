"""Tests for the embedders and the smart default.

Proves:
  1. OpenAIEmbedder calls the (injected) client and returns the vector.
  2. FunctionEmbedder wraps any callable (local models).
  3. default_embedder falls back to HashingEmbedder + WARNS when no OpenAI key is set.
"""
from __future__ import annotations

import types

import pytest

from coalent.semantic import FunctionEmbedder, HashingEmbedder, OpenAIEmbedder
from coalent.semantic.embedding import default_embedder


class _FakeOpenAI:
    """Minimal stand-in for openai.OpenAI used by OpenAIEmbedder (bring-your-own-client)."""

    def __init__(self) -> None:
        self.embeddings = self

    def create(self, *, model, input):  # type: ignore[no-untyped-def]
        return types.SimpleNamespace(data=[types.SimpleNamespace(embedding=[0.1, 0.2, 0.3])])


def test_openai_embedder_uses_injected_client() -> None:
    emb = OpenAIEmbedder(client=_FakeOpenAI())
    assert emb.embed("hello") == [0.1, 0.2, 0.3]


def test_function_embedder_wraps_a_callable() -> None:
    emb = FunctionEmbedder(lambda text: [float(len(text))])
    assert emb.embed("ab") == [2.0]


def test_default_embedder_falls_back_to_hashing_without_key(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.warns(UserWarning, match="HashingEmbedder"):
        emb = default_embedder()
    assert isinstance(emb, HashingEmbedder)


def test_openai_embed_many_chunks_at_provider_input_limit() -> None:
    # The OpenAI Embeddings API rejects >2048 inputs per request (hard provider
    # limit): embed_many must split transparently, order preserved — measured live:
    # an oversized evidence-sentence hydration batch 400-failed before this guard.
    class _CountingEmbeddings:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def create(self, *, model, input):  # type: ignore[no-untyped-def]
            assert len(input) <= 2048
            self.batch_sizes.append(len(input))
            return types.SimpleNamespace(data=[
                types.SimpleNamespace(embedding=[float(i)]) for i in range(len(input))])

    class _Client:
        embeddings = _CountingEmbeddings()

    client = _Client()
    emb = OpenAIEmbedder(client=client)
    out = emb.embed_many([f"t{i}" for i in range(5000)])
    assert len(out) == 5000
    assert client.embeddings.batch_sizes == [2048, 2048, 904]
    assert emb.embed_many([]) == []
