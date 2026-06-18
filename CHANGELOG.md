# Changelog

All notable changes to this project are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

## [0.2.1]

### Added
- **`OpenAIEmbedder`** — real semantic embeddings via the OpenAI Embeddings API (extra `coalent[openai]`); bring your own client or let it read `OPENAI_API_KEY`. Defaults to `text-embedding-3-small` (`text-embedding-3-large` for max accuracy).
- **`FunctionEmbedder`** — wrap any `text -> sequence[float]` callable (e.g. a local sentence-transformers model) as an `Embedder`.

### Changed
- **Smart default embedder.** When no `embedder` is passed, `SemanticCache` now auto-uses `OpenAIEmbedder` if the `openai` package is installed and `OPENAI_API_KEY` is set — accurate, semantic cache matching out of the box. Otherwise it falls back to the lexical `HashingEmbedder` **and emits a warning** (keyword-overlap matching can miss semantically-similar queries). Always overridable with `embedder=`.

## [0.2.0]

First public release — a real-time, provenance-invalidated cognitive cache for AI
agents and RAG. Framework-neutral, pluggable, and fully typed.

### Added
- **SemanticCache** — a `get(query)` read path: an embedding-keyed cache of
  decision-ready understanding that retains the raw evidence with every unit, so it
  can never return less than plain retrieval.
- **Provenance invalidation** — each unit records the exact sources it used;
  `source_changed` / `source_deleted` dirty only the units that used them and skip
  no-op changes via a content-hash compare. Units re-materialize lazily on read.
- **Retrievers** — `InMemoryRetriever`, `FunctionRetriever`, `CompositeRetriever`, a
  `BaseVectorRetriever`, and shipped bring-your-own-client adapters for Qdrant,
  Chroma, and pgvector (with pass-through search).
- **Synthesizers** — `LLMSynthesizer` (structured, citation-grounded understanding;
  user-owned instruction + fields) and `JSONPassthroughSynthesizer` (cache already
  structured tool/API JSON with no LLM call). Providers for OpenAI, Anthropic, and a
  deterministic stub.
- **Context intelligence** — coverage gate with auto-escalation, minimum-context
  projection, context strategies, and lazy cross-unit relationships.
- **Freshness** — `FreshnessPolicy` (TTL + revalidate-by-hash) for feed-less
  API/tool sources.
- **Persistence** — durable, restart-safe stores: `InMemoryCognitionStore`,
  `SQLiteCognitionStore` (stdlib, no server), and `RedisCognitionStore`. The
  invalidation graph rebuilds on startup.
- **Events** — GitHub / deploy / Jira / generic-CDC connectors with HMAC signature
  verification and an event dispatcher.
- **Integrations** — a graph-node helper (`make_cognition_node`) and MCP tool specs
  (`build_mcp_tools`) over the one-call read API.
- **CLI** — `coalent` inspects and manages a persistent cache (ls / show / dirty /
  invalidate / evict / purge / stats).
- **Eval harness** — measures accuracy, stale-rate, and token cost against naive RAG
  and a no-invalidation cache.

### Notes
- The core installs with **zero required dependencies**. Optional extras add LLM
  providers (`openai`, `anthropic`), vector adapters (`qdrant`, `chroma`,
  `pgvector`), the distributed store (`redis`), and a webhook server (`server`).
- Alpha: the public API may change before 1.0.
