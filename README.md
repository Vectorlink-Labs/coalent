<p align="center">
  <img src="https://raw.githubusercontent.com/Vectorlink-Labs/coalent/main/brand/wordmark.png" alt="Coalent" width="320">
</p>

<p align="center">
  <b>Real-time, provenance-invalidated context for AI agents &amp; RAG.</b><br>
  <i>Build understanding once. Reuse it everywhere. Keep it fresh — automatically.</i>
</p>

<p align="center">
  <img alt="pypi" src="https://img.shields.io/pypi/v/coalent?color=5145E5">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-4F46E5">
  <img alt="license" src="https://img.shields.io/badge/license-Apache%202.0-22D3EE">
  <img alt="typed" src="https://img.shields.io/badge/mypy-strict-2DD4BF">
  <img alt="tests" src="https://img.shields.io/badge/tests-passing-10B981">
  <a href="https://discord.gg/v3hvg3nwr"><img alt="discord" src="https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white"></a>
</p>

<p align="center">
  <b>📖 <a href="https://coalent.ai/docs">Documentation</a></b> &nbsp;·&nbsp; <a href="https://coalent.ai">coalent.ai</a> &nbsp;·&nbsp; <a href="https://discord.gg/v3hvg3nwr">💬 Discord</a>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> ·
  <a href="#whats-new-in-v05">What's new in v0.5</a> ·
  <a href="#the-read-path--a-ladder-of-gates">Gate ladder</a> ·
  <a href="#bring-your-own-stack">Bring your own stack</a> ·
  <a href="#benchmark">Benchmark</a> ·
  <a href="#cli">CLI</a>
</p>

---

> **Your agent re-reads the same sources on every call — and the moment a source changes, every cached answer is silently wrong.**
>
> Coalent builds the *understanding* once, caches it by what the query **means**, and invalidates it **surgically** the instant an underlying source changes. As correct as re-reading everything, at a fraction of the cost — and never stale.

## Why Coalent

Every context layer is forced to trade off three things. Coalent is built to hold all three at once:

- 🧠 **Extractive understanding, not chunks.** It caches a *query-independent* set of atomic, source-grounded **claims** your LLM extracted — keeping every number and fact — so one cached unit answers many *different* later questions. The raw evidence is retained with each unit, so a hit that under-covers a query falls back to retrieval instead of answering thin.
- ♻️ **Reuse across queries, agents — and documents.** A semantic cache keyed by query *meaning*: ask again, or from another agent, and it's a warm hit. **Cross-unit recall** pools claims across units to answer **multi-hop** questions whose evidence spans documents — at **zero extra LLM calls**.
- 🌿 **Fresh by provenance.** Every unit remembers the exact sources it used. When one changes, only the units that actually used it go stale — precisely, automatically, and lazily.

Coalent sits **above retrieval** — bring any retriever (vector DB, hybrid search, GraphRAG, tools, APIs). It's the freshness-and-reuse layer, not another retriever — deliberately the *opposite* of GraphRAG's build-the-whole-graph-upfront tax: **lightweight, independent units, built lazily only when a query actually needs one**, and refreshed by dirtying a single unit (no graph surgery).

> **New in v0.5** — `preset="multi_hop"`, source **widening**, provenance **admission**, a self-calibrating **adaptive** hit gate, `fast="auto"` numpy acceleration, structured **observability events**, and an experimental **pool serving** preview of the v0.6 read path (measured: statistically ties naive dense RAG's best arm at **0.79× its tokens**, pre-registered held-out n=605). All additive, all default-OFF. See [What's new](#whats-new-in-v05).

## Install

```bash
pip install coalent          # the core has zero required dependencies
```

## Quickstart

Runs as-is — `StubSynthesizer` needs no API key, so you can feel the loop in ten seconds:

```python
from coalent import SemanticCache, InMemoryRetriever, StubSynthesizer

# 1. Any retriever — a vector DB, a tool, an API. (In-memory here for the demo.)
retriever = InMemoryRetriever()
retriever.add("confluence:hr", "Leave policy: 21 days of annual leave per year.")

# 2. Build the cache. Swap StubSynthesizer for a real LLM below.
cache = SemanticCache(retriever, StubSynthesizer())

# 3. Ask. The first call builds understanding and caches it; the next is a warm hit.
result = cache.get("what is our leave policy?")
print(result.context["understanding"])
print(result.cache_hit)        # False (cold) -> True on the next call

# 4. A source changed? Only the units that used it go stale — surgically.
cache.source_changed("confluence:hr", text="Leave policy: now 25 days.")
# the next matching read rebuilds just that one unit, lazily
```

Wire in a real model — any text-in / text-out LLM works. In v0.4 the synthesizer builds **extractive** understanding by default (query-independent atomic claims that keep every fact), and the cache does **cross-unit recall** — both on automatically:

```python
from coalent import SemanticCache, LLMSynthesizer, OpenAIProvider, OpenAIEmbedder

cache = SemanticCache(
    retriever,
    LLMSynthesizer(OpenAIProvider(), model="gpt-4o-mini"),   # extract=True by default (v0.4)
    embedder=OpenAIEmbedder(),   # match queries by MEANING (recommended for real use)
)
# Multi-hop across documents? recall is already on; raise its trigger to bridge units:
#   SemanticCache(retriever, synth, embedder=..., recall_threshold=0.7)
```

## What's new in v0.5

The pool release — everything a month-long, pre-registered benchmark war on real news data
(MultiHopRAG, 609 articles, third-party questions) taught us, shipped as opt-in features:

- **`preset="multi_hop"`** — one argument arms cross-unit recall + the hop-2 bridge with
  calibrated thresholds. Explicit kwargs always win.
- **Source widening** (`widen_chunks=24`) — a miss-triggered build reads up to N chunks of the
  dominant source instead of only the retrieved keyhole. Effect in E2E: rebuild churn 460 → 31,
  warm-pass accuracy flipped from decaying to compounding. Never fires at ingest.
- **Provenance admission** (`provenance_admission=True`) — an exact-text containment probe
  prevents duplicate understanding: covered reads serve without building; thin coverage
  widen-rebuilds in place.
- **Adaptive hit gate** (`adaptive_hit=True`) — self-calibrates against score inflation as the
  cache grows (fixed thresholds provably absorb everything at scale).
- **Pool serving preview** (`serve="pool"`, `serve_budget=600`, `pool_header=...`) — serve the
  token-budgeted, globally ranked fresh-claim pool instead of one routed unit (**experimental**;
  the v0.6 read path). Held-out n=605: 0.699 accuracy vs 0.579 for unit serving (z=6.66);
  statistically ties naive dense RAG's best measured arm at 0.79× its tokens; 95% null honesty.
  Stale units' claims are masked from the pool the moment a source changes.
- **`fast="auto"`** — numpy-accelerated read path when numpy is present
  (`pip install "coalent[fast]"`); results are equivalence-pinned to the pure-Python core.
- **Observability** (`on_event=...`) — structured freshness events: builds, rebuilds, admission
  reuse, stale reads prevented, recall and bridge activity.
- Deprecated: `select_floor` (superseded by pool serving).

Full numbers and method in the [benchmark](#benchmark) section and CHANGELOG.

## What's new in v0.4

Two capabilities that were an opt-in preview are now the **defaults**, because they're strictly better on the structured / reuse-heavy corpora Coalent targets — and free or dormant everywhere else. Both have a one-line escape hatch back to exact v0.3 (`extract=False`, `cross_unit_recall=False`).

- 🎯 **Extractive understanding (`extract=True`, default).** Instead of a question-shaped prose summary, the synthesizer extracts a *query-independent* list of atomic, source-grounded claims. The same unit now answers many *different* later questions, and **no number is dropped** — a prose summary silently lost ~40% of the numbers in a source in our tests.
- 🔗 **Cross-unit claim recall (`cross_unit_recall=True`, default).** When one unit under-covers a query, the cache pools per-claim memory across **all** fresh units (MaxSim) and surfaces the bridge facts — answering **multi-hop** questions naive retrieval *structurally can't* (evidence in a document that doesn't resemble the question), at **zero extra LLM calls**. Dormant/free on single-hop; auto-off under a non-semantic embedder. Surfaced as `result.recalled`.
- 🛡️ **Precision & serving knobs (opt-in, default off):** `hit_margin` (refuse ambiguous ties), `select_floor` (serve atoms by meaning, fewer tokens), `residual_floor` (recover extractor-missed number spans). See the [gate ladder](#the-read-path--a-ladder-of-gates) for when to reach for each.

Upgrading from v0.3? See **[UPGRADE-0.3-to-0.4.md](UPGRADE-0.3-to-0.4.md)** — additive, one behaviour change (understanding is now claims, not prose).

## How it works

```
        query ──► embed ──► semantic cache
                               │  hit & fresh?  ──► serve cached understanding  (no retrieval, no LLM)
                               │  miss / stale? ─┐
                               ▼                 ▼
                          your Retriever ──► your Synthesizer ──► Cognition unit
                          (vector/tool/API)   (LLM or passthrough)  { understanding
                               ▲                                      + raw evidence
                               │                                      + provenance }
   source changed ────────────┘   dirties ONLY the units that used that source
```

1. **Embed the query** and look for an existing unit with similar meaning.
2. **Hit + fresh** → return the cached understanding (no retrieval, no LLM call).
3. **Miss or stale** → retrieve, synthesize understanding, **retain the raw evidence**, record **provenance** (the exact sources used), and cache it.
4. **A source changes** → `source_changed(id)` marks only the units whose provenance includes that id; they rebuild lazily on the next read.

Unchanged content is skipped via a content-hash compare, so a no-op change costs nothing.

## The read path — a ladder of gates

Coalent keys on what a unit **knows** — an embedding of its *understanding*, not the query's words — so *"how many vacation days?"* hits your leave unit, while *"exchange policy"* does **not**. Every `get(query)` then walks a fixed ladder of gates. The defaults are **pure cosine** — no extra model, no heavy dependency — and each gate is a tunable knob. In firing order:

| # | Gate | Default | Fires when → what happens |
|---|---|---|---|
| 1 | **`hit_threshold`** — match | auto (OpenAI ~0.33) | best unit's blended score (`0.7·topic + 0.3·seed`) below it → **miss** → retrieve + synthesize a new unit |
| 2 | **`hit_margin`** — precision guard | `0.0` (off) | top unit beats runner-up by less than the margin → ambiguous → build the query's own unit instead |
| 3 | **freshness** | provenance / TTL | matched unit dirty or expired → re-materialize it |
| 4 | **coverage** — does it answer? | max per-claim cosine | how well the matched unit covers *this* query (one perfect claim = covered) |
| 5 | **`cross_unit_recall`** | **on (v0.4)** | coverage `< recall_threshold` → pool the best claims across **all** fresh units (MaxSim), can lift coverage. Free when dormant, no LLM call |
| 6 | **`coverage_scorer` (S2)** | `None` (off) | in the ambiguous band `[coverage_floor, coverage_ceiling)` → a cross-encoder / NLI / LLM entailment check overrides cosine |
| 7 | **`coverage_floor`** — the RAG floor | auto (~0.28) | coverage still below it → **escalate**: append fresh raw retrieval (no LLM call), so a thin hit falls back to retrieval rather than answering wrong |
| 8 | **`select_floor`** — serve | `None` (lexical trim) | serve the unit's atoms by *meaning* (per-claim cosine ≥ floor) instead of a keyword trim — the query-relevant facts, fewer tokens |

Plus one **build-time** knob — `residual_floor`: retain number-bearing source spans the extractor dropped (best per-claim cosine < floor) as extra atoms. Embedding-only.
Other hooks: `route_by_claim` (late-interaction routing over a fat unit's claims), `relevance_gate` (BYO reranker before synthesis), `depth` (synthesis completeness vs cost), `calibrate_thresholds()` / `suggest_thresholds()`.

**Which knob for which workload** — the defaults are tuned for structured, single-hop reuse; reach for these when your data differs:

| Reach for… | When |
|---|---|
| **`recall_threshold ≈ 0.7`** | **multi-hop / cross-document** questions — makes recall bridge partially-covered reads (the full multi-hop win) |
| **`hit_margin > 0`** | **contradiction- / collision-heavy** corpora where near-ties are ambiguous (costs rebuilds — leave off on clean data) |
| **`select_floor`** | **paraphrase-heavy** queries over large units (a keyword trim misses when query and claim share no words) |
| **`residual_floor`** | **messy real prose** where the extractor might drop a number (cheap insurance) |
| **`coverage_scorer` (S2)** | **high-stakes ambiguity** where a wrong serve is costly (adds one judge call per borderline read) |

`stats()` reports `hit_rate`, `escalation_rate`, and the active thresholds, so you can see — and tune — exactly what the cache is doing.

## Bring your own stack

Coalent owns a tiny contract and passes everything else through to your tools.

**Retrievers** — a ladder from one-liner to full control:

| You have… | Use |
|---|---|
| Qdrant / Chroma / pgvector | a shipped adapter (bring-your-own-client) |
| another vector DB | extend `BaseVectorRetriever` |
| an existing search function | `FunctionRetriever` |
| several sources to fuse | `CompositeRetriever` |
| anything else | implement `Retriever` (one method) |

```python
from coalent import QdrantRetriever

retriever = QdrantRetriever(client=my_client, collection="docs", embed=my_embed)
```

**Synthesizers** — turn evidence into understanding:

- `LLMSynthesizer` — structured, citation-grounded understanding via your LLM (OpenAI, Anthropic, or any provider). You own the `instruction` and `fields`; Coalent owns the source / strict-JSON / citation envelope, so provenance is captured no matter what you ask for.
- `JSONPassthroughSynthesizer` — for already-structured tool/API JSON: caches it *as* the understanding, **no LLM call**.

**Embeddings** — how the cache matches queries by *meaning*. With `coalent[openai]` installed and `OPENAI_API_KEY` set, the cache uses OpenAI embeddings **automatically**; otherwise it warns and falls back to a lexical matcher. Override anytime:

```python
from coalent import SemanticCache, OpenAIEmbedder, FunctionEmbedder

cache = SemanticCache(retriever, synthesizer, embedder=OpenAIEmbedder("text-embedding-3-large"))
# or a local model: embedder=FunctionEmbedder(lambda t: my_model.encode(t).tolist())
```

> Use a real embedder for semantic matching — the no-key `HashingEmbedder` fallback matches on keyword overlap, not meaning, so similar-but-differently-worded queries can miss the cache.

**Stores** — durable and restart-safe (the invalidation graph rebuilds on startup):

```python
from coalent import SemanticCache, SQLiteCognitionStore   # stdlib, no server
from coalent import RedisCognitionStore                   # shared across processes / hosts

cache = SemanticCache(retriever, synthesizer, store=SQLiteCognitionStore("coalent.db"))
```

**Any agent framework** — the read API is a single call, so it drops in anywhere. Shipped helpers for graph nodes and MCP tools:

```python
from coalent import make_cognition_node, build_mcp_tools

node = make_cognition_node(cache)     # a graph node: state -> { context: fresh understanding }
tools = build_mcp_tools(cache)        # expose the cache as an MCP tool
```

## Benchmark

### Real-world: MultiHopRAG (v0.5, pre-registered)

609 real news articles, third-party gold questions, answered by gpt-4.1-mini with exact-match
grading — the corpus **maximally friendly to chunk retrieval** (questions are generated from
article sentences), chosen as the adversarial test. We run the fairness control most benchmarks
skip: **naive's own token-scaling curve** on the same stream (k4 0.58 @ 590 tok · k6 0.64 @ 882 ·
k9 0.71 @ 1311, n=605 held-out).

- **Pool serving (`serve="pool"`, warmed cache): 0.699 @ ~1,036 tokens** — beats naive k6
  (paired McNemar z=3.22) and **statistically ties naive's best measured arm at 0.79× its
  tokens** (z=0.60). We do not claim to beat the curve here; the claim is match-at-fewer-tokens
  plus what retrieval alone cannot do (freshness, provenance, compounding reuse).
- **Null honesty** (n=100 unanswerable): pool **95%** refusal vs naive's 85–88%.
- **Build layer** (cold-start, on-the-fly): widened units read a median **23 chunks** of their
  source vs 2 for keyhole builds; rebuild churn **460 → 31**; warm-pass accuracy flipped from
  decaying (−0.03) to compounding (+0.04).
- Misattribution 2–6%; cross-unit recall fired on ~90% of reads (fully instrumented).

### Structured regime (synthetic templates, v0.4)

Measured honestly on the structured / reuse workload Coalent is built for — **64 sources × 3 seeds = 192 reads per condition**, real OpenAI embeddings, a **deterministic** number-and-attribute accuracy check (no LLM-judge self-preference), and a **real dense top-5 retriever shared by both arms** (the naive RAG baseline *is* that retriever). Accuracy is graded escalation-off, so a fallback can't launder a win.

**Same accuracy as naive RAG, at a fraction of the context tokens** — across four answer models (95% CIs overlap on every model):

| Answer model | Naive RAG | Coalent v0.4 |
|---|:---:|:---:|
| gpt-4o-mini  | 0.81 | 0.81 |
| gpt-4.1-mini | 0.90 | 0.85 |
| gpt-4o       | 0.90 | 0.87 |
| gpt-4.1      | 0.99 | 0.97 |
| **Context tokens / read** | **126** | **47** |

And on the metrics that decide whether a cache is *trustworthy*, not just cheap:

- 🎯 **Routing — `route@1 ≈ 1.00`.** The cache picks the correct source unit essentially every time.
- 🛡️ **Misattribution — `~0–2%`.** How often it serves a number from the *wrong* source — the same noise floor as naive RAG's own answerer. (An earlier "27%" traced back to a benchmark bug — contradictory duplicate sources no router can resolve; found, fixed, documented. See the [transparency note](https://coalent.ai/docs/benchmark).)
- 🔗 **Multi-hop — naive `0%` → Coalent `100%`.** On bridge questions whose second-hop evidence doesn't resemble the question, single-shot retrieval answers **0%**; cross-unit recall answers **100%**, at **zero extra LLM calls**.
- 💰 **Economics — build once, reuse cheaply.** Understanding costs ~430 tokens / ~4s to build per source (once), then every later read is a warm cosine hit at ~⅓ the context. **Break-even ≈ 4–5 reads per source** — cheaper forever after.

*Full per-model and per-knob breakdown, methodology, and the benchmark-transparency note (what we found, fixed, and how) in the [docs](https://coalent.ai/docs/benchmark).*

## CLI

Installing Coalent gives you a `coalent` command — a `redis-cli` for your cognition cache (over a SQLite store):

```console
$ coalent ls
STATUS  HITS  AGE SRC  ID                  QUERY
fresh      6   2m   2  cog:c95a9d2897e0af  what is our leave policy?
dirty      1  12m   1  cog:7f1a0b9c3d2e4f  remote work rules

$ coalent show cog:c95a9d2897e0af      # understanding + provenance + raw evidence
$ coalent invalidate confluence:98231  # fire a change event
$ coalent stats
```

## Documentation

📚 **Full docs: [coalent.ai/docs](https://coalent.ai/docs)** — concepts, provenance & freshness, retrievers, synthesizers, persistence, worked examples (vector search, MCP & tools, agents), and the complete `get()` / data-model reference.

## Install options

```bash
pip install coalent                 # core, zero required deps
pip install "coalent[openai]"       # OpenAI provider      (also: anthropic)
pip install "coalent[qdrant]"       # vector adapters      (also: chroma, pgvector)
pip install "coalent[redis]"        # distributed store
pip install "coalent[dev]"          # tests + lint + types
```

## Contributing

Issues and PRs welcome. Run the gate before pushing:

```bash
pip install -e ".[dev]"
pytest && ruff check src && mypy src
```

## Status &amp; license

**Alpha** — the API may change before 1.0. Fully typed (`mypy --strict`), linted, and tested.

Licensed under [Apache-2.0](./LICENSE).

<p align="center"><sub>Context that's trustworthy, not just cheap.</sub></p>
