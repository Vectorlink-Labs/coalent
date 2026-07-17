"""The embedding-keyed semantic cognitive cache.

The read path: ``get(query)`` only, units that retain their raw evidence, and
provenance-driven, skip-no-op invalidation.
"""
from __future__ import annotations

from .cache import (
    PRESETS,
    ContextStrategy,
    FreshnessPolicy,
    InvalidationResult,
    Related,
    Result,
    SemanticCache,
)
from .calibrate import calibrate_thresholds, suggest_thresholds
from .embedding import (
    Embedder,
    FunctionEmbedder,
    HashingEmbedder,
    OpenAIEmbedder,
    cosine,
    default_thresholds_for,
)
from .memory import (
    BaseVectorRetriever,
    CompositeRetriever,
    FunctionRetriever,
    InMemoryRetriever,
    StubSynthesizer,
)
from .ports import Chunk, Generation, LLMProvider, Retriever, Synthesis, Synthesizer, Usage
from .store import (
    CognitionStore,
    InMemoryCognitionStore,
    RedisCognitionStore,
    SQLiteCognitionStore,
)
from .synthesizer import EXTRACTIVE_INSTRUCTION, JSONPassthroughSynthesizer, LLMSynthesizer
from .unit import Cognition
from .vector import ChromaRetriever, PgVectorRetriever, QdrantRetriever

__all__ = [
    "PRESETS",
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
    "Usage",
    "Generation",
    "LLMProvider",
    "LLMSynthesizer",
    "EXTRACTIVE_INSTRUCTION",
    "JSONPassthroughSynthesizer",
    "Embedder",
    "HashingEmbedder",
    "OpenAIEmbedder",
    "FunctionEmbedder",
    "cosine",
    "default_thresholds_for",
    "calibrate_thresholds",
    "suggest_thresholds",
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
