"""The embedding-keyed semantic cognitive cache.

The read path: ``get(query)`` only, units that retain their raw evidence, and
provenance-driven, skip-no-op invalidation.
"""
from __future__ import annotations

from .cache import (
    ContextStrategy,
    FreshnessPolicy,
    InvalidationResult,
    Related,
    Result,
    SemanticCache,
)
from .embedding import Embedder, FunctionEmbedder, HashingEmbedder, OpenAIEmbedder, cosine
from .memory import (
    BaseVectorRetriever,
    CompositeRetriever,
    FunctionRetriever,
    InMemoryRetriever,
    StubSynthesizer,
)
from .ports import Chunk, LLMProvider, Retriever, Synthesis, Synthesizer
from .store import (
    CognitionStore,
    InMemoryCognitionStore,
    RedisCognitionStore,
    SQLiteCognitionStore,
)
from .synthesizer import JSONPassthroughSynthesizer, LLMSynthesizer
from .unit import Cognition
from .vector import ChromaRetriever, PgVectorRetriever, QdrantRetriever

__all__ = [
    "SemanticCache",
    "ContextStrategy",
    "FreshnessPolicy",
    "Result",
    "Related",
    "InvalidationResult",
    "Cognition",
    "Chunk",
    "Retriever",
    "Synthesizer",
    "Synthesis",
    "LLMProvider",
    "LLMSynthesizer",
    "JSONPassthroughSynthesizer",
    "Embedder",
    "HashingEmbedder",
    "OpenAIEmbedder",
    "FunctionEmbedder",
    "cosine",
    "InMemoryRetriever",
    "FunctionRetriever",
    "CompositeRetriever",
    "BaseVectorRetriever",
    "QdrantRetriever",
    "ChromaRetriever",
    "PgVectorRetriever",
    "StubSynthesizer",
    "CognitionStore",
    "InMemoryCognitionStore",
    "SQLiteCognitionStore",
    "RedisCognitionStore",
]
