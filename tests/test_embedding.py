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
