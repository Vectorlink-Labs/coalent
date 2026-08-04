"""The refusal loop, runnable and fully offline — LangGraph-shaped, zero langgraph dep.

The pattern (Coalent's behavioral residual-span fallback, driven from LangChain):

    retrieve ──> synthesize ──(refused?)──> report_refusal ──> re-synthesize ──> report_success
        ^                └─(answered)──> done                        └─(still refused)──> done

A refusal over a served payload is EVIDENCE that extraction lost a fact. Handing the
read's ``read_id`` back via ``cache.report_refusal()`` returns the verbatim source
excerpts the extraction missed (tier-2 residual spans); a successful retry confirmed
via ``cache.report_success()`` durably attaches the refused query as an alternate
key on the owning unit (``key_confirmed``) — the cache learns from its own misses.

This script is a plain conditional loop implementing the identical pattern.
``langgraph`` is deliberately NOT a dependency of langchain-coalent; the 1:1
StateGraph mapping is::

    graph = StateGraph(State)
    graph.add_node("retrieve", retrieve)              # CoalentRetriever.invoke
    graph.add_node("synthesize", synthesize)          # your answerer over the payload
    graph.add_node("recover", recover)                # cache.report_refusal(read_id)
    graph.add_node("confirm", confirm)                # cache.report_success(read_id)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_conditional_edges("synthesize", is_refusal,
                                {True: "recover", False: "confirm"})
    graph.add_edge("recover", "synthesize")           # re-synthesize with the excerpts
    graph.add_edge("confirm", END)

Run:  python examples/refusal_loop.py
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from typing import Any, Callable

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.vectorstores import VectorStore
from pydantic import Field

from langchain_coalent import CoalentRetriever, create_coalent_cache

# --------------------------------------------------------------------------- fakes
# Offline stand-ins for the user's real stack (their vector DB, their embeddings,
# their chat model). Swap these three classes for e.g. Chroma + OpenAIEmbeddings +
# ChatOpenAI and the loop below is unchanged.

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and the of for to in on at by with how many much does do is are was were "
    "what which who it this that or as be".split()
)


class BagOfWordsEmbeddings(Embeddings):
    """Deterministic hashed bag-of-words embeddings (cosine == token overlap)."""

    def __init__(self, dim: int = 8192) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _TOKEN.findall(text.lower()):
            if tok in _STOP or len(tok) < 2:
                continue
            if len(tok) > 3 and tok.endswith("s"):
                tok = tok[:-1]
            digest = hashlib.md5(tok.encode("utf-8")).digest()
            vec[int.from_bytes(digest[:4], "big") % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class TinyVectorStore(VectorStore):
    """A minimal in-memory LangChain VectorStore."""

    def __init__(self, embedding: Embeddings) -> None:
        self._embedding = embedding
        self._docs: list[Document] = []
        self._vectors: list[list[float]] = []

    def add_texts(
        self,
        texts: Any,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        texts = list(texts)
        for i, text in enumerate(texts):
            self._docs.append(
                Document(page_content=text, metadata=dict(metadatas[i]) if metadatas else {})
            )
            self._vectors.append(self._embedding.embed_documents([text])[0])
        return [f"doc-{i}" for i in range(len(texts))]

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> "TinyVectorStore":
        store = cls(embedding)
        store.add_texts(texts, metadatas, ids=ids)
        return store

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        qv = self._embedding.embed_query(query)
        order = sorted(
            range(len(self._docs)),
            key=lambda i: -sum(a * b for a, b in zip(qv, self._vectors[i])),
        )
        return [self._docs[i] for i in order[:k]]


class ScriptedChatModel(BaseChatModel):
    """A scripted fake BaseChatModel: ``script(full_prompt_text) -> reply``."""

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
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.script(prompt)))]
        )


# ------------------------------------------------------------------- the scenario
# One HR document. The scripted "extractor" model captures the vacation fact but
# LOSES the 45-day relocation deadline (the measured harmful-extraction-loss mode
# residual spans exist for). The scripted "answerer" model refuses unless the
# deadline is actually in its context — no fabrication.

DOC = (
    "The standard vacation allowance is 21 days per year for full-time staff. "
    "Employees must file the relocation form 45 days before the relocation date."
)

LOSSY_CLAIMS = [
    "The standard vacation allowance is 21 days per year for full-time staff.",
    "Staff moving abroad must submit a relocation request to HR in advance.",  # number LOST
]

QUESTION = "Staff moving abroad must submit the relocation request how many days in advance?"


def extractor_script(prompt: str) -> str:
    return json.dumps({
        "summary": "Vacation and relocation policy facts.",
        "claims": LOSSY_CLAIMS,
        "entities": ["vacation", "relocation"],
        "facts": {},
        "used": [0],
    })


def answerer_script(prompt: str) -> str:
    if "45 days" in prompt:
        return "File the relocation form 45 days before the relocation date."
    return "REFUSE: the provided context does not state the deadline."


# ------------------------------------------------------------------ the graph run
def main() -> int:
    embeddings = BagOfWordsEmbeddings()
    vectorstore = TinyVectorStore(embeddings)
    vectorstore.add_texts([DOC], metadatas=[{"source": "policy:hr"}])
    extractor = ScriptedChatModel(script=extractor_script)
    answerer = ScriptedChatModel(script=answerer_script)

    events: list[dict[str, Any]] = []
    cache = create_coalent_cache(
        vectorstore,
        llm=extractor,
        embeddings=embeddings,
        residual_spans=True,     # the tier-2 span side channel (opt-in)
        query_keys=True,         # learn confirmed alternate keys (opt-in, pool path)
        serve_gate=0.30,         # explicit gate: exact, reproducible offline demo
        on_event=events.append,
    )
    retriever = CoalentRetriever(cache=cache)

    def synthesize(payload: str) -> str:
        reply = answerer.invoke(
            [SystemMessage(content="Answer ONLY from the provided context."),
             HumanMessage(content=f"CONTEXT:\n{payload}\n\nQUESTION: {QUESTION}")]
        )
        return str(reply.content)

    # retrieve ------------------------------------------------------------------
    doc = retriever.invoke(QUESTION)[0]
    read_id, payload = doc.metadata["read_id"], doc.page_content
    print(f"[retrieve]    read_id={read_id} sources={doc.metadata['sources']}")
    print(f"[payload]     {payload!r}")

    # synthesize -> conditional refusal edge ------------------------------------
    answer = synthesize(payload)
    print(f"[synthesize]  {answer!r}")
    refused_first = answer.startswith("REFUSE")
    assert refused_first, "demo invariant: extraction lost the deadline, so this must refuse"

    # report_refusal -> re-synthesize -------------------------------------------
    retry_payload = cache.report_refusal(read_id)
    assert retry_payload is not None, "demo invariant: the residual span must qualify"
    print(f"[recover]     report_refusal -> {retry_payload!r}")
    answer = synthesize(payload + "\n\n" + retry_payload)
    print(f"[re-synth]    {answer!r}")
    assert not answer.startswith("REFUSE"), "retry must succeed with the recovered excerpt"
    assert "45 days" in answer

    # report_success ------------------------------------------------------------
    cache.report_success(read_id)
    names = [e.get("event") for e in events]
    print(f"[confirm]     events={names}")
    assert "residual_fallback" in names
    assert "key_confirmed" in names, "the confirmed retry must durably attach the key"

    print("refusal loop complete: refused -> recovered '45 days' -> key_confirmed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
