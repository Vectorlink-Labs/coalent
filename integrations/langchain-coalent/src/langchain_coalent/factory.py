"""``create_coalent_cache`` — the one-call constructor.

Wraps your LangChain vector store (or retriever), adapts your LangChain
embeddings and chat model into Coalent's ``Embedder``/``Synthesizer`` ports,
applies the recommended defaults, and passes every Coalent knob through.
"""
from __future__ import annotations

from typing import Any

from coalent import (
    Cognition,
    Embedder,
    HashingEmbedder,
    LLMSynthesizer,
    SemanticCache,
    StubSynthesizer,
    Synthesizer,
)
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from .adapters import (
    ChatModelProvider,
    CoalentVectorStoreRetriever,
    LangChainEmbedder,
    chat_model_name,
)

#: kwargs consumed by the vector-store adapter rather than the cache constructor.
_ADAPTER_KNOBS = ("k", "search_kwargs")
#: kwargs consumed by the LLMSynthesizer envelope rather than the cache constructor.
_SYNTH_KNOBS = ("extract", "instruction", "fields", "depth")


def default_pool_header(unit: Cognition) -> str:
    """The recommended pool attribution header: ``[{artifact_id}]`` from the
    unit's first evidence chunk (falls back to the unit's seed query, then id).
    Pass your own ``pool_header=`` for richer ``[title | source | date]``
    attribution when your metadata has it."""
    artifact = unit.evidence[0].artifact_id if unit.evidence else ""
    if artifact:
        return f"[{artifact}]"
    query = (unit.query or "").strip()
    return ("## " + query[:60]) if query else f"[source: {unit.id}]"


def create_coalent_cache(
    vectorstore_or_retriever: VectorStore | BaseRetriever | Any,
    llm: BaseChatModel | None = None,
    embeddings: Embeddings | None = None,
    **knobs: Any,
) -> SemanticCache:
    """Build a :class:`~coalent.SemanticCache` over your LangChain stack in one call.

    Args:
        vectorstore_or_retriever: any LangChain ``VectorStore`` or ``BaseRetriever``
            (wrapped via :class:`CoalentVectorStoreRetriever`), or any object already
            implementing Coalent's ``Retriever`` protocol (used as-is — full BYO).
        llm: a LangChain ``BaseChatModel`` used, as you configured it, for synthesis
            (query-independent extractive units by default). ``None`` falls back to
            Coalent's deterministic ``StubSynthesizer`` — fine for wiring tests, not
            for production understanding. A custom Coalent ``Synthesizer`` can be
            passed via ``synthesizer=`` instead.
        embeddings: a LangChain ``Embeddings`` object keying the cache semantically
            (adapted via :class:`LangChainEmbedder`). ``None`` uses Coalent's default
            (OpenAI when configured, else the lexical hashing fallback).
        **knobs: every ``SemanticCache`` knob passes through unchanged
            (``hit_threshold``, ``key_floor``, ``serve_budget``, ``preset``, ...),
            plus adapter knobs ``k``/``search_kwargs`` (vector-store search) and
            synthesizer-envelope knobs ``extract``/``instruction``/``fields``/``depth``.

    Recommended defaults applied (each only when you did not set it):

    * ``read_path="pool"`` — the v0.6 measured operating point — whenever you
      supplied a semantic embedder (``embeddings=``, or a non-hashing
      ``embedder=``). Otherwise the factory stays on ``read_path="unit"``: the
      pool path requires a semantic embedder by contract, and without an explicit
      one the factory won't guess.
    * ``pool_header=default_pool_header`` on the pool path, so payloads carry
      per-source attribution instead of the bare-header fallback.
    * Behavioral knobs (``residual_spans``, ``query_keys``, ...) stay opt-in.
    """
    # --- retriever: BYO protocol first, else wrap the LangChain object -------------
    adapter_kwargs = {name: knobs.pop(name) for name in _ADAPTER_KNOBS if name in knobs}
    if isinstance(vectorstore_or_retriever, (VectorStore, BaseRetriever)):
        retriever: Any = CoalentVectorStoreRetriever(vectorstore_or_retriever, **adapter_kwargs)
    elif callable(getattr(vectorstore_or_retriever, "retrieve", None)):
        if adapter_kwargs:
            raise TypeError(
                "k/search_kwargs configure the LangChain vector-store adapter; "
                f"{type(vectorstore_or_retriever).__name__} is already a Coalent Retriever"
            )
        retriever = vectorstore_or_retriever
    else:
        raise TypeError(
            "vectorstore_or_retriever must be a langchain_core VectorStore, a BaseRetriever, "
            f"or a Coalent Retriever; got {type(vectorstore_or_retriever).__name__}"
        )

    # --- embedder ------------------------------------------------------------------
    if embeddings is not None and "embedder" in knobs:
        raise TypeError("pass either embeddings= (LangChain) or embedder= (Coalent), not both")
    embedder: Embedder | None
    if embeddings is not None:
        embedder = LangChainEmbedder(embeddings)
    elif "embedder" in knobs:
        embedder = knobs.pop("embedder")
    else:
        embedder = None  # let SemanticCache resolve its own default

    # --- synthesizer ---------------------------------------------------------------
    synth_kwargs = {name: knobs.pop(name) for name in _SYNTH_KNOBS if name in knobs}
    if llm is not None and "synthesizer" in knobs:
        raise TypeError("pass either llm= (LangChain chat model) or synthesizer=, not both")
    synthesizer: Synthesizer
    if llm is not None:
        synthesizer = LLMSynthesizer(
            ChatModelProvider(llm), model=chat_model_name(llm), **synth_kwargs
        )
    elif "synthesizer" in knobs:
        if synth_kwargs:
            raise TypeError(
                "extract/instruction/fields/depth configure the LLMSynthesizer envelope; "
                "they need llm=, not a custom synthesizer"
            )
        synthesizer = knobs.pop("synthesizer")
    else:
        if synth_kwargs:
            raise TypeError(
                "extract/instruction/fields/depth configure the LLMSynthesizer envelope; "
                "pass llm= to use them"
            )
        synthesizer = StubSynthesizer()

    # --- recommended defaults (explicit knobs always win) --------------------------
    # The pool read path requires a semantic embedder BY CONTRACT, so it defaults on
    # only when the caller supplied one we can vouch for (embeddings= or a non-hashing
    # embedder=). With neither, the factory cannot know what Coalent's default resolves
    # to, so it conservatively stays on the unit path — pass embeddings= (or
    # read_path="pool") to run the recommended v0.6 operating point.
    semantic = embedder is not None and not isinstance(embedder, HashingEmbedder)
    knobs.setdefault("read_path", "pool" if semantic else "unit")
    if knobs["read_path"] == "pool":
        knobs.setdefault("pool_header", default_pool_header)

    if embedder is not None:
        knobs["embedder"] = embedder
    return SemanticCache(retriever, synthesizer, **knobs)
