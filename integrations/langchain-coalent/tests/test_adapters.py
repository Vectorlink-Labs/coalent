"""LG1 — adapter: LangChain Documents round-trip to Coalent Chunks with artifact ids,
plus the Embeddings and BaseChatModel port adapters against the real installed API."""
from __future__ import annotations

import json

import pytest
from conftest import BagOfWordsEmbeddings, FakeVectorStore, ScriptedChatModel
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from coalent import Chunk, Generation
from langchain_coalent import (
    ChatModelProvider,
    CoalentVectorStoreRetriever,
    LangChainEmbedder,
    chat_model_name,
    document_to_chunk,
)


class StaticRetriever(BaseRetriever):
    """A minimal LangChain BaseRetriever returning fixed documents."""

    docs: list[Document]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.docs


# --------------------------------------------------------------- document_to_chunk
def test_artifact_id_priority_explicit_wins() -> None:
    doc = Document("t", metadata={"artifact_id": "a:1", "source": "s:1", "id": "i:1"}, id="d:1")
    assert document_to_chunk(doc).artifact_id == "a:1"


def test_artifact_id_source_convention() -> None:
    doc = Document("t", metadata={"source": "confluence:98231", "id": "i:1"}, id="d:1")
    assert document_to_chunk(doc).artifact_id == "confluence:98231"


def test_artifact_id_document_id_outranks_metadata_id() -> None:
    doc = Document("t", metadata={"id": "i:1"}, id="d:1")
    assert document_to_chunk(doc).artifact_id == "d:1"


def test_artifact_id_metadata_id_fallback() -> None:
    doc = Document("t", metadata={"id": "i:1"})
    assert document_to_chunk(doc).artifact_id == "i:1"


def test_artifact_id_content_hash_fallback_is_deterministic() -> None:
    a, b = document_to_chunk(Document("same text")), document_to_chunk(Document("same text"))
    assert a.artifact_id == b.artifact_id
    assert a.artifact_id.startswith("chunk:")
    assert document_to_chunk(Document("other text")).artifact_id != a.artifact_id


def test_page_content_and_version_round_trip() -> None:
    doc = Document("Leave policy: 21 days.", metadata={"source": "hr", "version": 7})
    chunk = document_to_chunk(doc)
    assert isinstance(chunk, Chunk)
    assert chunk.text == "Leave policy: 21 days."
    assert chunk.version == "7"  # coerced to str per Chunk contract


# --------------------------------------------------- CoalentVectorStoreRetriever
def test_wraps_vectorstore(embeddings: BagOfWordsEmbeddings) -> None:
    vs = FakeVectorStore(embeddings)
    vs.add_texts(
        ["The leave allowance is 21 days.", "Travel goes through the portal."],
        metadatas=[{"source": "policy:leave"}, {"source": "policy:travel"}],
    )
    chunks = CoalentVectorStoreRetriever(vs).retrieve("how many days of leave allowance?")
    assert chunks and chunks[0].artifact_id == "policy:leave"
    assert chunks[0].text == "The leave allowance is 21 days."
    assert {c.artifact_id for c in chunks} == {"policy:leave", "policy:travel"}


def test_wraps_vectorstore_k_respected(embeddings: BagOfWordsEmbeddings) -> None:
    vs = FakeVectorStore(embeddings)
    vs.add_texts(["alpha fact one", "beta fact two", "gamma fact three"])
    assert len(CoalentVectorStoreRetriever(vs, k=1).retrieve("alpha fact")) == 1


def test_wraps_base_retriever() -> None:
    lc = StaticRetriever(docs=[Document("doc text", metadata={"source": "s:1"})])
    chunks = CoalentVectorStoreRetriever(lc).retrieve("anything")
    assert [(c.artifact_id, c.text) for c in chunks] == [("s:1", "doc text")]


def test_rejects_non_langchain_object() -> None:
    with pytest.raises(TypeError, match="VectorStore or"):
        CoalentVectorStoreRetriever(object())  # type: ignore[arg-type]


# ------------------------------------------------------------- LangChainEmbedder
def test_embedder_adapter_matches_langchain_sides(embeddings: BagOfWordsEmbeddings) -> None:
    adapter = LangChainEmbedder(embeddings)
    assert adapter.embed("leave policy") == embeddings.embed_query("leave policy")
    assert adapter.embed_many(["a b", "c d"]) == embeddings.embed_documents(["a b", "c d"])
    assert adapter.embed_many([]) == []


# ------------------------------------------------------------- ChatModelProvider
def test_chat_provider_returns_generation_with_usage() -> None:
    chat = ScriptedChatModel(script=lambda prompt: json.dumps({"ok": True}))
    gen = ChatModelProvider(chat).generate(
        model="fake-model", system="sys prompt", user="user prompt", max_tokens=64, temperature=0.0
    )
    assert isinstance(gen, Generation)
    assert gen.text == json.dumps({"ok": True})
    assert gen.usage is not None
    assert gen.usage.prompt_tokens > 0 and gen.usage.completion_tokens > 0
    assert gen.usage.model == "fake-model"
    # the chat model saw both roles, concatenated by the fake
    assert "sys prompt" in chat.calls[0] and "user prompt" in chat.calls[0]


def test_chat_model_name_fallback() -> None:
    chat = ScriptedChatModel(script=lambda prompt: "x")
    assert chat_model_name(chat) == "ScriptedChatModel"
