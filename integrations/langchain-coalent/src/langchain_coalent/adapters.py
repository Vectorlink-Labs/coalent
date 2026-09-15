"""Port adapters: LangChain objects in, Coalent ports out.

BYO-first by construction — your existing LangChain index, embeddings, and chat
model become the substrate of a :class:`coalent.SemanticCache` without changing
how you built any of them:

  * :class:`CoalentVectorStoreRetriever` — any LangChain ``VectorStore`` or
    ``BaseRetriever`` as a Coalent ``Retriever`` (the evidence substrate).
  * :class:`LangChainEmbedder` — any LangChain ``Embeddings`` as a Coalent
    ``Embedder`` (the cache's semantic key space).
  * :class:`ChatModelSynthesizer` via :class:`ChatModelProvider` — any LangChain
    ``BaseChatModel`` as the cache's synthesis engine.
"""
from __future__ import annotations

import hashlib
from typing import Any

from coalent import Chunk, Generation, Usage
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

#: Metadata keys tried, in order, when mapping a Document to a Chunk artifact id.
_ARTIFACT_METADATA_KEYS = ("artifact_id", "source", "id")

#: Metadata keys copied into ``Chunk.meta`` when present (v0.7 ingest metadata —
#: Coalent's built-in pool header renders them as ``[{title} | {source} | {date}]``).
_META_KEYS = ("title", "source", "date")


def document_to_chunk(doc: Document) -> Chunk:
    """Map one LangChain ``Document`` to a Coalent ``Chunk``.

    ``page_content`` becomes ``Chunk.text``. The artifact id — the natural source
    identity Coalent's provenance invalidation keys on — resolves in this order:

    1. ``metadata["artifact_id"]`` (explicit, always wins)
    2. ``metadata["source"]`` (the LangChain loader convention)
    3. ``Document.id`` (the typed id field)
    4. ``metadata["id"]``
    5. fallback: ``"chunk:" + sha1(page_content)[:12]`` — content-derived and
       deterministic, so re-retrieving identical text maps to the same artifact.
       With the fallback, ``source_changed`` can only target text you re-supply
       verbatim; give your documents a ``source`` for real invalidation.

    ``metadata["version"]`` (if present) becomes ``Chunk.version``. The v0.7 ingest-
    metadata keys ``title``/``source``/``date`` (if present) are copied into
    ``Chunk.meta`` — Coalent's built-in pool attribution header serves them as
    ``[{title} | {source} | {date}]``. When ``source`` is absent but another meta
    key is present, it falls back to the resolved artifact id (the natural source
    identity) — never to the content-derived ``chunk:`` digest, and never alone:
    a doc with NO meta keys maps to ``meta=None`` so the header ladder keeps its
    measured query-title rung instead of downgrading to an opaque id line.
    Remaining metadata is not carried on the chunk (best-effort mapping, documented).
    """
    artifact_id = ""
    for key in _ARTIFACT_METADATA_KEYS:
        value = doc.metadata.get(key)
        if value is not None and str(value).strip():
            artifact_id = str(value)
            break
        if key == "source" and doc.id:  # typed Document.id outranks metadata["id"]
            artifact_id = str(doc.id)
            break
    if not artifact_id:
        digest = hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()[:12]
        artifact_id = f"chunk:{digest}"
    meta: dict[str, str] = {}
    for key in _META_KEYS:
        value = doc.metadata.get(key)
        if value is not None and str(value).strip():
            meta[key] = str(value)
    if meta and "source" not in meta and not artifact_id.startswith("chunk:"):
        meta["source"] = artifact_id
    version = doc.metadata.get("version")
    return Chunk(
        artifact_id=artifact_id,
        text=doc.page_content,
        version="" if version is None else str(version),
        meta=meta or None,
    )


class CoalentVectorStoreRetriever:
    """Wrap any LangChain ``VectorStore`` OR ``BaseRetriever`` as a Coalent ``Retriever``.

    The BYO bridge: your existing index — unchanged — becomes the cache's evidence
    substrate. A ``VectorStore`` is searched via ``similarity_search(query, k=k,
    **search_kwargs)``; a ``BaseRetriever`` via ``invoke(query)`` (its own ``k``/
    search config governs). Returned ``Document``s map to ``Chunk``s per
    :func:`document_to_chunk`.

    ``namespace`` (Coalent's optional tenant scope) is accepted but not forwarded:
    LangChain has no portable namespace concept. To scope searches, encode the
    filter in ``search_kwargs`` (VectorStore) or in the retriever you wrap.
    """

    def __init__(
        self,
        vs_or_retriever: VectorStore | BaseRetriever,
        *,
        k: int = 4,
        search_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(vs_or_retriever, (VectorStore, BaseRetriever)):
            raise TypeError(
                "CoalentVectorStoreRetriever wraps a langchain_core VectorStore or "
                f"BaseRetriever; got {type(vs_or_retriever).__name__}"
            )
        self._wrapped = vs_or_retriever
        self._k = k
        self._search_kwargs = dict(search_kwargs or {})

    def retrieve(self, query: str, *, namespace: str | None = None) -> list[Chunk]:
        del namespace  # documented: no portable LangChain namespace concept
        if isinstance(self._wrapped, VectorStore):
            docs = self._wrapped.similarity_search(query, k=self._k, **self._search_kwargs)
        else:
            docs = self._wrapped.invoke(query)
        return [document_to_chunk(doc) for doc in docs]


class LangChainEmbedder:
    """Adapt a LangChain ``Embeddings`` object to Coalent's ``Embedder`` port.

    ``embed`` uses ``embed_query`` (the query-side vector); ``embed_many`` uses
    ``embed_documents`` — one batched call, which Coalent's per-claim embedding
    exploits automatically.
    """

    def __init__(self, embeddings: Embeddings) -> None:
        self._embeddings = embeddings

    def embed(self, text: str) -> list[float]:
        return [float(v) for v in self._embeddings.embed_query(text)]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [[float(v) for v in vec] for vec in self._embeddings.embed_documents(list(texts))]


def _message_text(message: BaseMessage) -> str:
    """Extract plain text from a chat message across langchain-core 0.3 and 1.x.

    ``content`` may be a plain string (classic) or a list of content blocks
    (1.x standard content). We read ``content`` directly instead of ``.text``
    because ``.text`` is a method in 0.3 and a property in 1.x.
    """
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


class ChatModelProvider:
    """Adapt a LangChain ``BaseChatModel`` to Coalent's ``LLMProvider`` port.

    BYO means *your model, your configuration*: the provider invokes the chat
    model exactly as you configured it — the ``model`` / ``max_tokens`` /
    ``temperature`` arguments Coalent's ``LLMSynthesizer`` passes are NOT
    forwarded (there is no portable kwarg contract across LangChain chat
    integrations; e.g. ``max_tokens`` vs ``max_output_tokens``). Configure
    determinism (temperature 0 is recommended for the strict-JSON synthesis
    contract) on the chat model itself.

    Token usage is read from ``AIMessage.usage_metadata`` when the integration
    reports it, so ``Result.usage`` and ``cache.stats()`` keep working.
    """

    def __init__(self, chat_model: BaseChatModel) -> None:
        self._chat_model = chat_model

    def generate(
        self, *, model: str, system: str, user: str, max_tokens: int, temperature: float
    ) -> Generation:
        del max_tokens, temperature  # documented: the chat model's own config governs
        message = self._chat_model.invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        usage_metadata = getattr(message, "usage_metadata", None)
        usage: Usage | None = None
        if usage_metadata:
            usage = Usage(
                prompt_tokens=int(usage_metadata.get("input_tokens", 0) or 0),
                completion_tokens=int(usage_metadata.get("output_tokens", 0) or 0),
                model=model or chat_model_name(self._chat_model),
            )
        return Generation(text=_message_text(message), usage=usage)


def chat_model_name(chat_model: BaseChatModel) -> str:
    """Best-effort model label for usage accounting (``model_name``/``model`` attrs)."""
    for attr in ("model_name", "model"):
        value = getattr(chat_model, attr, None)
        if isinstance(value, str) and value:
            return value
    return type(chat_model).__name__
