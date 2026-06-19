"""coalent — a real-time, provenance-invalidated cognitive cache for AI.

Coalent caches the *understanding* your LLM builds, keyed by query meaning (a
semantic cache), keeps the raw evidence with every unit, and invalidates surgically
by provenance as sources change — so context stays cheap, fresh, and reusable across
queries and agents. It is the freshness/reuse layer ABOVE retrieval: bring any
retriever (vector DB, GraphRAG, tools, APIs).

Quickstart::

    from coalent import SemanticCache, InMemoryRetriever, StubSynthesizer

    retriever = InMemoryRetriever()
    retriever.add("confluence:hr-handbook", "Leave policy: 21 days of annual leave...")

    cache = SemanticCache(retriever, StubSynthesizer())
    ctx = cache.get("what is our leave policy?")     # ask by meaning — just the query
    cache.source_changed("confluence:hr-handbook", text="Leave policy: now 25 days...")
"""
from __future__ import annotations

# --- The cache (primary API) ---
from .semantic import (
    BaseVectorRetriever,
    Chunk,
    ChromaRetriever,
    Cognition,
    CognitionStore,
    CompositeRetriever,
    ContextStrategy,
    Embedder,
    FreshnessPolicy,
    FunctionEmbedder,
    FunctionRetriever,
    HashingEmbedder,
    InMemoryCognitionStore,
    InMemoryRetriever,
    InvalidationResult,
    JSONPassthroughSynthesizer,
    LLMProvider,
    LLMSynthesizer,
    OpenAIEmbedder,
    PgVectorRetriever,
    QdrantRetriever,
    RedisCognitionStore,
    Related,
    Result,
    Retriever,
    SemanticCache,
    SQLiteCognitionStore,
    StubSynthesizer,
    Synthesis,
    Synthesizer,
    calibrate_thresholds,
    cosine,
    default_thresholds_for,
    suggest_thresholds,
)

# --- Shared primitives (provenance + change events) ---
from .domain.models import ChangeEvent, ProvenanceManifest, SourceSpan, Status

# --- Integrations & infrastructure ---
from .integrations import build_mcp_tools, make_cognition_node
from .adapters.providers import AnthropicProvider, OpenAIProvider, StubProvider
from .events import (
    DeployConnector,
    EventDispatcher,
    GenericCDCConnector,
    GitHubConnector,
    JiraConnector,
    verify_github_signature,
)

__version__ = "0.3.0"

__all__ = [
    "__version__",
    # --- The cache ---
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
    # --- primitives ---
    "SourceSpan",
    "ProvenanceManifest",
    "ChangeEvent",
    "Status",
    # --- infrastructure ---
    "EventDispatcher",
    "GitHubConnector",
    "DeployConnector",
    "JiraConnector",
    "GenericCDCConnector",
    "verify_github_signature",
    "StubProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    # --- integrations ---
    "make_cognition_node",
    "build_mcp_tools",
]
