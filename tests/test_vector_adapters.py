"""Vector adapters: bring-your-own-client mapping + capability detection, via fakes."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from coalent.semantic import ChromaRetriever, PgVectorRetriever, QdrantRetriever


def _embed(_text: str) -> list[float]:
    return [0.1, 0.2, 0.3]


class _QdrantModern:
    """A client exposing the modern `query_points` API."""

    def query_points(self, **kwargs: Any) -> Any:
        points = [SimpleNamespace(payload={"artifact_id": "confluence:98231", "text": "leave", "version": "7"})]
        return SimpleNamespace(points=points)


class _QdrantLegacy:
    """A client exposing only the legacy `search` API."""

    def search(self, **kwargs: Any) -> list[Any]:
        return [SimpleNamespace(payload={"artifact_id": "jira:OPS-1", "text": "ticket", "version": "2"})]


def test_qdrant_capability_detection_modern() -> None:
    retriever = QdrantRetriever(client=_QdrantModern(), collection="docs", embed=_embed)
    chunks = retriever.retrieve("leave policy")
    assert chunks[0].artifact_id == "confluence:98231"
    assert chunks[0].version == "7"


def test_qdrant_capability_detection_legacy() -> None:
    retriever = QdrantRetriever(client=_QdrantLegacy(), collection="docs", embed=_embed)
    chunks = retriever.retrieve("a ticket")
    assert chunks[0].artifact_id == "jira:OPS-1"  # used the legacy search() path


def test_qdrant_passthrough_search_override() -> None:
    from coalent import Chunk

    def my_search(query: str, namespace: str | None) -> list[Any]:
        return [SimpleNamespace(payload={"artifact_id": f"doc:{query}", "text": query})]

    retriever = QdrantRetriever(client=object(), collection="docs", search=my_search)
    assert retriever.retrieve("hybrid")[0].artifact_id == "doc:hybrid"
    assert isinstance(retriever.retrieve("x")[0], Chunk)


def test_chroma_columnar_mapping() -> None:
    class _Collection:
        def query(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "ids": [["chunk-1"]],
                "documents": [["remote work policy"]],
                "metadatas": [[{"artifact_id": "confluence:44120", "version": "3"}]],
            }

    retriever = ChromaRetriever(collection=_Collection(), embed=_embed)
    chunk = retriever.retrieve("remote work")[0]
    assert chunk.artifact_id == "confluence:44120"
    assert chunk.version == "3"


def test_pgvector_via_search_override() -> None:
    def my_search(query: str, namespace: str | None) -> list[Any]:
        return [{"id": "db:orders/42", "text": "order 42", "version": "v9"}]

    retriever = PgVectorRetriever(connection=object(), table="docs", search=my_search)
    chunk = retriever.retrieve("order status")[0]
    assert chunk.artifact_id == "db:orders/42"
    assert chunk.version == "v9"
