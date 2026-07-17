# Changelog

All notable changes to this project are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

## [0.5.0]

The pool release. v0.5 ships the results of a month-long benchmark war on real news data
(MultiHopRAG, 609 articles, third-party gold questions): the build path now reads whole
sources instead of retrieval keyholes, admission prevents duplicate understanding by
provenance, the hit gate self-calibrates as the cache grows — and an experimental preview
of the v0.6 read path serves the globally ranked claim pool instead of one routed unit.
Every number below is pre-registered and measured on held-out questions.

### Added
- `preset="multi_hop"` — one argument arms cross-unit recall + the hop-2 bridge with
  calibrated thresholds. Explicit kwargs always override a preset.
- `serve="pool"` + `serve_budget` + `pool_header` (**experimental**, preview of the v0.6
  read path): serve the token-budgeted, globally ranked fresh-claim pool instead of the
  routed unit's claims. Measured (held-out n=605, real news): 0.699 accuracy @ ~1,040
  context tokens vs 0.579 for the unit path (McNemar z=6.66); statistically ties naive
  dense RAG's best measured configuration at 0.79x its tokens; 95% null honesty (vs
  naive's 85-88%). Stale units' claims are masked from the pool the moment a source
  changes — freshness is preserved by construction.
- Source widening — `widen_chunks`, `source_fetcher`, `widen_on_admission`: a
  miss-triggered build reads up to N chunks of the dominant source (duck-typed
  `retriever.widen(artifact_id, limit=)` or your `source_fetcher`) instead of only the
  retrieved keyhole. E2E effect: rebuild churn 460 -> 31; warm-pass accuracy flipped from
  decaying (-0.03) to compounding (+0.04). Lazy covenant intact: never fires at ingest.
- Provenance admission — `provenance_admission`: before paying for a build, an exact-text
  containment probe checks whether retrieval's chunks are already inside fresh units;
  covered reads serve without building, thin coverage widen-rebuilds in place.
- `split_by_artifact`: one unit per source when retrieval mixes artifacts.
- `adaptive_hit` + seed-reuse channel: the hit gate self-calibrates against cross-unit
  score inflation as the cache grows (fixed thresholds provably absorb everything at
  scale); repeat/paraphrase queries keep hitting via the seed channel.
- `recall_bridge` + `bridge_limit`: hop-2 bridge restart (rank other units' claims by
  similarity to the matched unit's own claims — where hop-2 lives when it does not
  resemble the question). Armed by `preset="multi_hop"`.
- `on_event` observability hook + structured freshness events (`unit_built`,
  `unit_rebuilt`, `admission_reuse`, `admission_widen_rebuild`, `stale_read_prevented`,
  `claims_recalled`, `bridge_claims`, `source_changed`).
- `fast="auto"` — numpy-accelerated read path (`pip install "coalent[fast]"`): the three
  O(cache-size) scans run as vectorized twins with identical control flow; equivalence is
  pinned by CI tests to float precision. Auto-detects numpy; the pure-Python fallback
  keeps the core zero-dependency.
- `Result.needs_retrieval` hint: the cache still under-covers after recall — your answerer
  may want fresh retrieval (an affordance, not an automatic action).

### Changed
- `select_floor` is deprecated (superseded by pool serving) and will be removed in a
  future release.

### Honesty notes
- All additions are opt-in / default-OFF (or `auto` with proven-identical results): v0.4
  code behaves identically on v0.5.
- Full regime map: on the adversarial open-domain news benchmark, pool serving matches
  naive dense RAG's best measured arm at 0.79x its tokens (statistical tie, n=605
  pre-registered) — we do not claim to beat it there. The structured/reuse-heavy regime
  keeps its ~0.33x parity result. Multi-hop-beyond-top-K and churn benchmarks are tracked
  for v0.6.

## [0.4.0]

Cognition units, on by default. v0.4 turns the seed query into **query-independent
extractive understanding** and adds **cross-unit claim recall** — and, unlike an opt-in
preview, both are now the **defaults**, because they are strictly better on the structured /
reuse-heavy corpora Coalent targets. A prose summary silently drops facts (in our tests
~40% of the numbers in a source); extractive units keep them all and let one cached unit
answer many later questions. Recall lets the cache answer multi-hop questions that span
documents — at zero extra LLM calls. Both have a one-line escape hatch back to exact v0.3.

### Added
- **Extractive understanding** — `LLMSynthesizer(extract=True)` builds a query-INDEPENDENT
  list of atomic, source-grounded claims instead of a question-shaped prose summary, so the
  same unit serves many different later questions and no number is lost. Now the default;
  pass `extract=False` for the v0.3 prose path. Exposes `EXTRACTIVE_INSTRUCTION`.
- **Cross-unit claim recall** — `SemanticCache(cross_unit_recall=True)` pools per-claim
  memory across ALL fresh units and surfaces the best-matching claims (MaxSim) when the
  single matched unit under-covers a query — recovering a bridge fact that lives in another
  unit (multi-hop) with no extra LLM call, only cosine over cached claims. Dormant (free) on
  single-hop and auto-disabled under a non-semantic embedder. `recall_threshold` controls
  when it fires (defaults to `coverage_floor`); `recall_limit` bounds the pool. Now the
  default; pass `cross_unit_recall=False` to restore v0.3.
- **`hit_margin`** — refuse to commit to a unit that only ties a topical neighbour by less
  than the margin (materialise the query's own unit instead). Opt-in precision guard for
  ambiguous / collision-heavy corpora.
- **`select_floor`** — serve the matched unit's atoms by MEANING (per-claim cosine ≥ floor)
  rather than the lexical keyword trim: the query-relevant facts, fewer tokens. Opt-in.
- **`residual_floor`** — at build time, retain number-bearing source spans the extractor
  dropped (best per-claim cosine < floor) as extra atoms, closing the extractor-recall gap
  on messy prose. Embedding-only, no extra LLM call. Opt-in (`residual_limit` bounds it).
- **`Usage` / `Generation`** surfaced on the synthesizer port for read-cost accounting.

### Changed
- **Defaults flipped ON: `extract=True` and `cross_unit_recall=True`.** They are strictly
  better on structured / reuse workloads and free-or-dormant elsewhere, so they now ship on.
  Upgrading changes the shape of cached understanding (atomic claims, not prose). To keep the
  exact v0.3 read behaviour, pass `extract=False` and `cross_unit_recall=False` — see
  [UPGRADE-0.3-to-0.4.md](UPGRADE-0.3-to-0.4.md). The situational knobs (`hit_margin`,
  `select_floor`, `residual_floor`, `coverage_scorer`) stay OFF by default.

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
- **Pluggable coverage scorer + two-tier** — cosine over per-claim embeddings stays the
  zero-dep default; an optional `coverage_scorer(query, understanding) -> float` (a
  cross-encoder / NLI / LLM entailment check) decides *containment* where cosine can't
  tell "adjacent topic" from "actually answers". `coverage_ceiling` consults the scorer
  only in the ambiguous band, so you pay for it on the few borderline queries, not every hit.
- **Per-claim routing (`route_by_claim`)** — optional late-interaction matching that routes
  a query by the unit's best-matching claim instead of its averaged understanding, so a
  query finds the unit holding a claim about it even inside a fat multi-claim unit.
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
