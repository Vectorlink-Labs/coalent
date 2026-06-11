"""Tests for the built-in retriever helpers (FunctionRetriever, BaseVectorRetriever)."""
from __future__ import annotations

from typing import Any

from coalent.semantic import BaseVectorRetriever, Chunk, FunctionRetriever


def test_function_retriever_wraps_a_callable() -> None:
    def search(query: str, namespace: str | None) -> list[Chunk]:
        return [Chunk(artifact_id=f"doc:{query}", text=f"about {query}")]

    retriever = FunctionRetriever(search)
    chunks = retriever.retrieve("leave")
    assert chunks[0].artifact_id == "doc:leave"


def test_base_vector_retriever_maps_hits_and_skips_empty() -> None:
    class MyVector(BaseVectorRetriever):
        def search(self, query: str, namespace: str | None) -> list[Any]:
            return [
                {"id": "98231", "text": "leave policy", "rev": "7"},
                {"id": "", "text": "ignore me"},  # no id -> skipped
            ]

        def to_chunk(self, hit: Any) -> Chunk | None:
            return Chunk(
                artifact_id=f"confluence:{hit['id']}" if hit["id"] else "",
                text=hit["text"],
                version=str(hit.get("rev", "")),
            )

    chunks = MyVector().retrieve("leave")
    assert len(chunks) == 1
    assert chunks[0].artifact_id == "confluence:98231"
    assert chunks[0].version == "7"
