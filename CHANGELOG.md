# Changelog

All notable changes to this project are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

## [0.3.0]

Understanding-keyed matching, semantic coverage, and tunable thresholds. The cache now
keys on what a unit **knows** (an embedding of its understanding), not the seed query —
so "exchange policy" no longer false-hits "leave policy", while genuine paraphrases
still hit. Coverage and escalation are semantic (per-claim), not lexical.

### Added
- **Understanding-keyed matching** — blends a *topic* score (query↔understanding
  embedding) with the *seed* score (query↔seed query), weighted by `understanding_weight`
  (default 0.7). Kills surface-form false hits; keeps paraphrase recall. Per-unit
  `understanding_embedding` + `claim_embeddings`, computed at build time (batched) and
  lazily backfilled for pre-0.3 units on load.
- **Semantic per-claim coverage + escalation** — a hit whose best per-claim cosine is
  below `coverage_floor` escalates to fresh raw for that query (still a hit), restoring
  the "never less than plain retrieval" floor *semantically*. Tunable via `coverage_floor`
  and the `enable_coverage_escalation` switch.
- **Embedder-aware default thresholds** — `hit_threshold` / `coverage_floor` derive from
  the embedder when unset (OpenAI ~0.33 vs lexical HashingEmbedder 0.6), so the OpenAI
  path works out of the box. Plus `calibrate_thresholds` (labeled) and `suggest_thresholds`
  (labels-free) helpers.
- **`relevance_gate`** — optional `(query, chunks) -> chunks` hook applied between retrieve
  and synthesize: de-noises the understanding, provenance, and the raw floor. BYO
  reranker / score threshold; Coalent never reranks itself.
- **Depth knob** — `LLMSynthesizer(depth=0.0..1.0)` trades synthesis cost against coverage
  completeness (terse / balanced / exhaustive).
- **`embed_many`** batch path on the embedders (one round-trip for K claims) via
  `embed_texts`, kept off the `Embedder` protocol so custom embedders still satisfy it.
- **Behavioral recording** — units remember the (bounded) queries that hit them
  (`hit_queries`).
- **Read observability** — `stats()` now reports `reads`, `hits`, `escalations`,
  `escalation_rate`, and `hit_rate` (the "am I drifting back to RAG?" signal).

### Changed
- `coverage_floor` is now a **semantic max-per-claim cosine** (was a lexical token-overlap
  fraction); `hit_threshold` now gates the **blended** score and auto-derives per embedder.
- `Cognition.touch()` gained an optional `query` argument (records `hit_queries`).
- The seed query/embedding no longer drift on re-materialization — a unit's identity is
  its understanding, not whichever query last rebuilt it.

### Removed
- The lexical coverage gate (`_coverage_over`) — replaced by semantic per-claim coverage.

## [0.2.1]

### Added
- **OpenAIEmbedder** and **FunctionEmbedder**, plus a smart `default_embedder` that
  auto-uses OpenAI embeddings when `coalent[openai]` is installed and `OPENAI_API_KEY` is
  set, otherwise falls back to the lexical `HashingEmbedder` with a warning — semantic
  cache hits out of the box on the recommended path.

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
