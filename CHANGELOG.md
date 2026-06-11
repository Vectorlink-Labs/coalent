# Changelog

All notable changes to this project are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

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
