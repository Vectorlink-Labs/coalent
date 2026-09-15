"""langchain-coalent — Coalent as a LangChain-native freshness/reuse layer.

BYO-first: your existing LangChain vector store (or retriever), embeddings, and
chat model become the substrate of a provenance-invalidated
:class:`~coalent.SemanticCache` — nothing about how you built them changes.

Quickstart::

    from langchain_coalent import CoalentRetriever, create_coalent_cache

    cache = create_coalent_cache(my_vectorstore, llm=my_chat_model,
                                 embeddings=my_embeddings)
    retriever = CoalentRetriever(cache=cache)      # drop-in LangChain retriever

    docs = retriever.invoke("what is our leave policy?")
    docs[0].page_content            # the served, attributed context payload
    docs[0].metadata["read_id"]     # -> cache.report_refusal() / report_success()

    cache.source_changed("policy.md", text=new_text)   # surgical invalidation
"""
from .adapters import (
    ChatModelProvider,
    CoalentVectorStoreRetriever,
    LangChainEmbedder,
    chat_model_name,
    document_to_chunk,
)
from .factory import create_coalent_cache, default_pool_header
from .retriever import CoalentRetriever, render_payload, sources_of

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "CoalentVectorStoreRetriever",
    "CoalentRetriever",
    "create_coalent_cache",
    "ChatModelProvider",
    "LangChainEmbedder",
    "chat_model_name",
    "document_to_chunk",
    "default_pool_header",
    "render_payload",
    "sources_of",
]
