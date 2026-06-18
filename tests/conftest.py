"""Shared test fixtures — keep the suite hermetic and deterministic.

The cache's default embedder now auto-uses OpenAI when OPENAI_API_KEY is set. Force
the deterministic, offline HashingEmbedder during tests so they never hit the network
or depend on a stray env key — and so the no-embedder warning doesn't spam output.
"""
from __future__ import annotations

import pytest

import coalent.semantic.cache as _cache
from coalent.semantic import HashingEmbedder


@pytest.fixture(autouse=True)
def _deterministic_embedder(monkeypatch):  # type: ignore[no-untyped-def]
    monkeypatch.setattr(_cache, "default_embedder", lambda: HashingEmbedder())
