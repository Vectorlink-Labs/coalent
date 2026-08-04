"""Offline-deterministic fakes for the LG gauntlet: a fake LangChain VectorStore,
deterministic fake embeddings, and a scripted fake chat model. No network, no API
keys — every score is exact bag-of-words arithmetic."""
from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Callable

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.vectorstores import VectorStore
from pydantic import Field

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and the of for to in on at by with how many much does do is are was were "
    "what which who it this that or as be".split()
)


def _tokens(text: str) -> list[str]:
    out: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in _STOP or len(tok) < 2:
            continue
        if len(tok) > 3 and tok.endswith("s"):
            tok = tok[:-1]
        out.append(tok)
    return out


class BagOfWordsEmbeddings(Embeddings):
    """Deterministic hashed bag-of-words embeddings (cosine == token overlap).

    Semantic enough for the gauntlet's exact arithmetic, fully offline. dim 8192
    keeps hash-bucket collisions negligible for the small test vocabularies.
    """

    def __init__(self, dim: int = 8192) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _tokens(text):
            digest = hashlib.md5(tok.encode("utf-8")).digest()
            vec[int.from_bytes(digest[:4], "big") % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FakeVectorStore(VectorStore):
    """Minimal in-memory LangChain VectorStore (cosine over fake embeddings).

    Mutable on purpose: ``replace_text`` simulates the user's ingestion pipeline
    updating a document in their index (the LG3 freshness scenario).
    """

    def __init__(self, embedding: Embeddings) -> None:
        self._embedding = embedding
        self._docs: list[Document] = []
        self._vectors: list[list[float]] = []

    # --- LangChain VectorStore contract -------------------------------------------
    def add_texts(
        self,
        texts: Any,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        texts = list(texts)
        out: list[str] = []
        for i, text in enumerate(texts):
            metadata = dict(metadatas[i]) if metadatas else {}
            doc_id = ids[i] if ids else None
            self._docs.append(Document(page_content=text, metadata=metadata, id=doc_id))
            self._vectors.append(self._embedding.embed_documents([text])[0])
            out.append(doc_id or f"fake-{len(self._docs) - 1}")
        return out

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> "FakeVectorStore":
        store = cls(embedding)
        store.add_texts(texts, metadatas, ids=ids)
        return store

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        qv = self._embedding.embed_query(query)
        scored = sorted(
            range(len(self._docs)),
            key=lambda i: -sum(a * b for a, b in zip(qv, self._vectors[i])),
        )
        return [self._docs[i] for i in scored[:k]]

    # --- test helper ----------------------------------------------------------------
    def replace_text(self, source: str, new_text: str) -> None:
        """Simulate the user's pipeline updating a document in their index."""
        for i, doc in enumerate(self._docs):
            if doc.metadata.get("source") == source:
                self._docs[i] = Document(page_content=new_text, metadata=dict(doc.metadata))
                self._vectors[i] = self._embedding.embed_documents([new_text])[0]
                return
        raise KeyError(source)


class ScriptedChatModel(BaseChatModel):
    """A scripted fake ``BaseChatModel``: ``script(full_prompt_text) -> reply``.

    Deterministic regardless of call order/count (unlike an iterator of canned
    messages), and reports ``usage_metadata`` so the usage plumbing is testable.
    """

    script: Callable[[str], str]
    calls: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = "\n".join(str(m.content) for m in messages)
        self.calls.append(prompt)
        reply = self.script(prompt)
        message = AIMessage(
            content=reply,
            usage_metadata={
                "input_tokens": len(prompt.split()),
                "output_tokens": len(reply.split()),
                "total_tokens": len(prompt.split()) + len(reply.split()),
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


@pytest.fixture()
def embeddings() -> BagOfWordsEmbeddings:
    return BagOfWordsEmbeddings()


@pytest.fixture()
def no_openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the offline gauntlet offline even on a machine with keys set."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
