"""Thin, version-agnostic vector-DB retrievers (Qdrant, Chroma, pgvector).

The philosophy, preserved from the original design:

  * **Bring your own client** — you pass the client you configured at the version
    you run. We never construct or pin it; vendor SDKs are imported lazily, so the
    base package installs without them.
  * **Pass-through search** — the default does a sensible top-k query, but pass your
    own ``search=`` callable to use hybrid search, custom filters, or reranking — any
    vendor feature, in full.
  * **Capability detection** — we use whichever API your client version exposes, so
    a client upgrade doesn't break the adapter.

These extend :class:`~coalent.semantic.memory.BaseVectorRetriever`: you get a
working retriever for free, or implement the :class:`~coalent.semantic.ports.Retriever`
protocol directly for total control.
"""
from __future__ import annotations

from typing import Any, Callable

from .memory import BaseVectorRetriever
from .ports import Chunk

EmbedFn = Callable[[str], Any]
SearchFn = Callable[[str, "str | None"], list[Any]]


class QdrantRetriever(BaseVectorRetriever):
    """A :class:`BaseVectorRetriever` backed by Qdrant."""

    def __init__(
        self,
        *,
        client: Any,
        collection: str,
        embed: EmbedFn | None = None,
        search: SearchFn | None = None,
        top_k: int = 6,
        id_field: str = "artifact_id",
        text_field: str = "text",
        version_field: str = "version",
        namespace_field: str = "namespace",
    ) -> None:
        self._client = client
        self._collection = collection
        self._embed = embed
        self._search_fn = search
        self._top_k = top_k
        self._id_field = id_field
        self._text_field = text_field
        self._version_field = version_field
        self._namespace_field = namespace_field

    def search(self, query: str, namespace: str | None) -> list[Any]:
        if self._search_fn is not None:
            return self._search_fn(query, namespace)  # full pass-through (hybrid, rerank…)
        if self._embed is None:
            raise ValueError("QdrantRetriever needs either `embed=` or a custom `search=` callable")
        vector = self._embed(query)
        query_filter = self._build_filter(namespace)
        # Capability detection across Qdrant client versions.
        if hasattr(self._client, "query_points"):
            response = self._client.query_points(
                collection_name=self._collection, query=vector, limit=self._top_k,
                query_filter=query_filter, with_payload=True,
            )
            return list(response.points)
        return list(
            self._client.search(
                collection_name=self._collection, query_vector=vector, limit=self._top_k,
                query_filter=query_filter, with_payload=True,
            )
        )

    def _build_filter(self, namespace: str | None) -> Any:
        if not namespace:
            return None
        from qdrant_client import models  # lazy: only the default filter path needs it

        return models.Filter(
            must=[models.FieldCondition(
                key=self._namespace_field, match=models.MatchValue(value=namespace),
            )]
        )

    def to_chunk(self, hit: Any) -> Chunk | None:
        payload = getattr(hit, "payload", None) or {}
        return Chunk(
            artifact_id=str(payload.get(self._id_field, "")),
            text=str(payload.get(self._text_field, "")),
            version=str(payload.get(self._version_field, "")),
        )


class ChromaRetriever(BaseVectorRetriever):
    """A :class:`BaseVectorRetriever` backed by a Chroma collection."""

    def __init__(
        self,
        *,
        collection: Any,
        embed: EmbedFn | None = None,
        search: SearchFn | None = None,
        top_k: int = 6,
        id_field: str = "artifact_id",
        text_field: str = "text",
        version_field: str = "version",
        namespace_field: str = "namespace",
    ) -> None:
        self._collection = collection
        self._embed = embed
        self._search_fn = search
        self._top_k = top_k
        self._id_field = id_field
        self._text_field = text_field
        self._version_field = version_field
        self._namespace_field = namespace_field

    def search(self, query: str, namespace: str | None) -> list[Any]:
        if self._search_fn is not None:
            return self._search_fn(query, namespace)
        where = {self._namespace_field: namespace} if namespace else None
        if self._embed is not None:
            result = self._collection.query(
                query_embeddings=[list(self._embed(query))], n_results=self._top_k, where=where,
            )
        else:
            result = self._collection.query(query_texts=[query], n_results=self._top_k, where=where)
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]
        hits = []
        for i, text in enumerate(documents):
            hits.append({
                "id": ids[i] if i < len(ids) else "",
                "text": text,
                "meta": metadatas[i] if i < len(metadatas) and metadatas[i] else {},
            })
        return hits

    def to_chunk(self, hit: Any) -> Chunk | None:
        meta = hit["meta"]
        return Chunk(
            artifact_id=str(meta.get(self._id_field) or hit["id"]),
            text=str(hit["text"]),
            version=str(meta.get(self._version_field, "")),
        )


class PgVectorRetriever(BaseVectorRetriever):
    """A :class:`BaseVectorRetriever` backed by Postgres + pgvector."""

    def __init__(
        self,
        *,
        connection: Any,
        table: str,
        embed: EmbedFn | None = None,
        search: SearchFn | None = None,
        top_k: int = 6,
        id_column: str = "artifact_id",
        text_column: str = "text",
        version_column: str = "version",
        embedding_column: str = "embedding",
        namespace_column: str = "namespace",
    ) -> None:
        self._conn = connection
        self._table = table
        self._embed = embed
        self._search_fn = search
        self._top_k = top_k
        self._id_column = id_column
        self._text_column = text_column
        self._version_column = version_column
        self._embedding_column = embedding_column
        self._namespace_column = namespace_column

    def search(self, query: str, namespace: str | None) -> list[Any]:
        if self._search_fn is not None:
            return self._search_fn(query, namespace)
        if self._embed is None:
            raise ValueError("PgVectorRetriever needs either `embed=` or a custom `search=` callable")
        vector = list(self._embed(query))
        where = f" WHERE {self._namespace_column} = %s" if namespace else ""
        sql = (
            f"SELECT {self._id_column}, {self._text_column}, {self._version_column} "
            f"FROM {self._table}{where} "
            f"ORDER BY {self._embedding_column} <=> %s::vector LIMIT %s"
        )
        params: list[Any] = ([namespace] if namespace else []) + [vector, self._top_k]
        with self._conn.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        return [{"id": r[0], "text": r[1], "version": r[2]} for r in rows]

    def to_chunk(self, hit: Any) -> Chunk | None:
        return Chunk(
            artifact_id=str(hit["id"]),
            text=str(hit["text"]),
            version=str(hit.get("version") or ""),
        )
