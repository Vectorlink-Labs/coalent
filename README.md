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
  <a href="#whats-new-in-v06">What's new in v0.6</a> ·
  <a href="#the-read-path--a-ladder-of-gates">Gate ladder</a> ·
  <a href="#bring-your-own-stack">Bring your own stack</a> ·
  <a href="#use-it-from-claude-code--cursor-mcp">MCP</a> ·
  <a href="#langchain">LangChain</a> ·
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

> **New in v0.6** — the **pool read path** (`read_path="pool"`): every read serves the token-budgeted, globally ranked fresh-claim pool. Measured on a 605-question news benchmark (strict grading): **0.731 accuracy @ 981 context tokens** — matching naive top-9 (0.711 @ 1,311) at **~25% fewer tokens**, and naive's best measured point (top-12: 0.731 @ 1,729) at **~43% fewer**. Plus a default-OFF **behavioral stack** — residual spans → refusal fallback → append-only repair → query keys — measured at **−33% refusals** and **+3.1 pts** on the same store. All opt-in; the default read path is unchanged v0.5 behavior. See [What's new](#whats-new-in-v06).
>
> **New in v0.6.1** — the **MCP server**: `coalent-mcp` puts the cache one line away from Claude Code, Cursor, or any MCP client ([Use it from Claude Code / Cursor](#use-it-from-claude-code--cursor-mcp)), and **[`langchain-coalent`](#langchain)** makes your existing LangChain stack the cache's substrate. Both additive-only.

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

**The v0.6 pool read path** — opt in, and every read serves the budget-packed, globally ranked fresh-claim pool instead of one routed unit. Attribution is the one thing to wire: a 3-line `pool_header` callable mapping each unit to `[title | source | date]` from your own corpus metadata. This is the measured golden path — on a 605-question news benchmark (strict grading), **0.68** accuracy with the bare built-in header vs **0.73** with this callable, same store, same queries:

```python
DOC_META = {  # your corpus metadata, keyed by artifact id
    "docs:azure-refresh": {"title": "Azure region refresh", "source": "CloudWire", "date": "2026-05-02"},
}

def pool_header(unit) -> str:   # the [title | source | date] golden path — 3 lines
    meta = DOC_META.get(unit.evidence[0].artifact_id if unit.evidence else "")
    return f"[{meta['title']} | {meta['source']} | {meta['date']}]" if meta else f"[source: {unit.id}]"

cache = SemanticCache(retriever, synthesizer, embedder=OpenAIEmbedder(),
                      read_path="pool", pool_header=pool_header)
result = cache.get("which regions got the refresh?")
result.context["pool"]   # the packed, attributed claim payload — hand it to your answer model
```

Runnable no-API-key demo, including the refusal loop: [examples/pool_read_path.py](examples/pool_read_path.py).

## Use it from Claude Code / Cursor (MCP)

<!-- mcp-name: io.github.nisarg-pujara-vectorlink/coalent -->

`coalent-mcp` serves fresh, attributed facts from a Coalent cache to any MCP client —
and the facts are invalidated the instant their source changes. One line to wire it into
Claude Code:

```bash
pip install "coalent[mcp,openai]"
claude mcp add coalent -- coalent-mcp --cache-factory my_cache:build
```

(Cursor / Claude Desktop / any MCP client: register the same `coalent-mcp ...` command in
its MCP config.)

**Bring your own cache (`--cache-factory module:function`) — the primary mode.** Your
factory returns a fully constructed `SemanticCache`: your vector DB, your embedder, your
LLM, every knob. The server adds protocol glue only — and the glue is measured to add
**zero quality loss**: factory mode reproduced the library's own benchmark result
byte-identically (0.710 on a 100-question validation run drawn from our n=605 news
benchmark — identical CIs, 100/100 serves, 98/100 answer payloads byte-equal to the
library run).

```python
# my_cache.py — importable from the directory you launch in
from coalent import (SemanticCache, LLMSynthesizer, OpenAIProvider,
                     OpenAIEmbedder, SQLiteCognitionStore)

def build() -> SemanticCache:
    return SemanticCache(
        my_vector_retriever,                  # YOUR vector DB / retriever
        LLMSynthesizer(OpenAIProvider()),     # YOUR synthesis model
        embedder=OpenAIEmbedder(),            # YOUR embedder
        read_path="pool",
        residual_spans=True, query_keys=True, # the behavioral stack, opt-in as ever
        pool_header=my_metadata_header,       # [title | source | date] — the measured golden path
        store=SQLiteCognitionStore("kb.db"),  # persistence is yours too
    )
```

Freshness here is signal-driven: your ingestion pipeline calls the `source_changed` tool
when a document changes and the affected facts invalidate immediately. (Adding
`--watch DIR` alongside the factory also fires it on file edits — invalidation only; it
never ingests into your index, and it matches only when your artifact ids equal the
watch-relative paths.)

**Zero-config folder mode (`--watch DIR`) — the demo wedge.** Point it at a folder of
docs and you get the recommended v0.6 deployment (pool path, residual spans, query keys,
SQLite persistence, automatic `[path | modified date]` attribution) with no code at all:

```bash
claude mcp add coalent --env OPENAI_API_KEY=$OPENAI_API_KEY -- coalent-mcp --watch ./docs
```

Every read rescans the watched files (mtime + content hash) before serving — you cannot
get a stale answer after saving a file — and an untouched folder restarts fully warm.
**The honest number:** on the same 100-question validation run, folder mode scored
**0.46 vs 0.71** for a factory-built cache (40 vs 18 refusals) — the measured cost of the
generic paragraph chunker and on-demand keyhole builds. Use it to feel the freshness loop
in a minute; bring your own stack for production quality. One regime note: a question
about a *just-added* file can honestly refuse from a warm cache until a read triggers
that file's first build — a refusal, never a stale or wrong answer.

**One shared cache for many agents (`--transport http`).**

```bash
COALENT_MCP_TOKEN=<secret> coalent-mcp --cache-factory my_cache:build --transport http --port 8765
```

One long-lived process, many concurrent MCP clients, ONE shared cache — shared
compounding, no store races (validated: two concurrent clients matched the sequential
reference on all 20 reads, zero duplicate builds). When `COALENT_MCP_TOKEN` is set, every
request must carry `Authorization: Bearer <token>` — bind localhost or trusted networks.
Corollary for stdio: each stdio launch is its own process, so never point two apps at the
same `--store` path — HTTP mode *is* the shared-cache answer.

**The seven tools:** `get_context(query, budget?)` → the attributed, budget-packed
payload + a `read_id` · `report_refusal(read_id)` / `report_success(read_id)` → the
behavioral repair loop over MCP · `source_changed(artifact_id, text?)` → the BYO
freshness feed (unchanged content is hash-detected and skipped) · `list_sources()` ·
`cache_stats()` · `refresh()`.

## LangChain

[`langchain-coalent`](integrations/langchain-coalent) makes Coalent a LangChain-native
freshness/reuse layer — BYO-first: your existing VectorStore (or retriever), embeddings,
and chat model become the cache's substrate, unchanged.

```bash
pip install langchain-coalent
```

```python
from langchain_coalent import create_coalent_cache, CoalentRetriever

cache = create_coalent_cache(my_vectorstore, llm=my_chat_model, embeddings=my_embeddings)
retriever = CoalentRetriever(cache=cache)      # drop-in LangChain BaseRetriever

docs = retriever.invoke("what is our leave policy?")
docs[0].page_content              # the served, attributed context payload
docs[0].metadata["read_id"]       # -> cache.report_refusal() / report_success()
docs[0].metadata["cache_hit"]     # True == served with zero LLM spend

cache.source_changed("policy.md", text=new_text)   # surgical, provenance-keyed invalidation
```

Every Coalent knob passes through `create_coalent_cache`; the refusal→repair loop ships
as a runnable LangGraph-shaped example in the package. Depends only on `coalent>=0.6` and
`langchain-core>=0.3`.

## What's new in v0.7

The self-healing release. The read now ships its own doubt, and the cache gains an
explicit failure chain your agent drives — each rung fires **only on a failed read**, so
a read that succeeds pays nothing new. **Everything is default-OFF and byte-inert until
armed** (pinned by tests): a 0.6 user who upgrades and touches nothing gets 0.6 behavior,
byte for byte. Full details in the [CHANGELOG](CHANGELOG.md).

<!-- PROPOSED — pending user sanction: every number in the next paragraph -->
Measured on the same frozen news rig as every anchor since v0.5 (609 articles, 605
held-out questions, strict grading + locked adjudication rules): the full v0.7
composition scores **0.826 vs 0.774** for the strongest v0.6 configuration — **+5.3
points at an identical ~983-token serving budget**, breakage 5.6% (under the rig's 9.4%
serving-order perturbation floor), final refusals **−69%** (61 → 19). Gating repair on
failure matches always-on accuracy at **14% of the extraction calls**.

The agentic loop — your evaluator decides, the cache heals:

```python
cache = SemanticCache(
    retriever, synthesizer, embedder=OpenAIEmbedder(),
    read_path="pool",
    gap_detector=True,               # observe-only: the read reports its own holes
    repair_extractor=my_extractor,   # BYO callable(span, region, existing_claims) -> [claims]
)

r = cache.get(
    question,
    subs=planner_subquestions,       # your planner's decomposition (optional)
    constraints={"dates": ["2023-10-05"], "sources": ["TechCrunch"]},  # intent metadata (optional)
)
answer = my_answerer(r.context["pool"])          # your model, your prompt
r.sources                                        # artifact ids behind the payload
r.max_source_age_s                               # freshness age of this serve
r.gaps                                           # [{probe, span, source, kind}, ...] — the read's own doubt

if failed(answer):                               # the failure chain — each rung ONLY on failure
    cache.repair(r.read_id)                      # re-extract what the build missed — PERMANENT
    r2 = cache.get(question, subs=planner_subquestions)   # repaired claims now compete
    answer = my_answerer(r2.context["pool"])
    if failed(answer):
        r3 = cache.reprobe(r2.read_id)           # entity-probe re-rank of the same pool
        answer = my_answerer(r3.context["pool"]) if r3 else answer
    if failed(answer):
        r4 = cache.serve_unserved(r2.read_id)    # last rung: force-pack admitted-but-unserved claims
        answer = my_answerer(r4.context["pool"]) if r4 else answer
```

- **The read ships its own provenance and doubt.** `Result.sources` (the artifact ids
  actually behind the payload), `Result.max_source_age_s` (the serve's freshness age),
  and — with `gap_detector=True` — `Result.probes` / `probe_coverage` / `gaps`: per
  probe, whether a raw evidence sentence outscores every claim (`extraction_hole` →
  repair terrain) or nothing reaches the probe (`corpus_hole` → route to a tool).
  Observe-only by construction: serving is byte-identical ON vs OFF (pinned), and the
  sentence tier costs $0 at serve (lazy, embedded once, cached).
- **`repair(read_id)` — the pump.** Consumes the read's banked gap and constraint
  candidates plus bridge candidates near what served; one span-anchored BYO extractor
  call per candidate (the library still never calls an LLM itself); mechanical near-dup
  and defect rejection (antecedent-free pronoun subjects, truncated spans); then an
  **append-only, provenance-stamped, permanent** store improvement — the cost is paid
  once, every future read benefits. Returns a `RepairReport`.
- **`serve_unserved(read_id)` and `reprobe(read_id, hint=None)` — the refusal rungs.**
  Force-pack the claims that lost the packing race, or re-rank the unchanged pool with
  entity probes harvested from what served. Embeds only — no LLM, no retrieval, no store
  mutation — and by contract they run on refusals, so they cannot un-answer a correct
  read.
- **`subs=` / `constraints=` on `get()`.** Your planner hands the read its decomposition
  (`subs=["...", ...]` — wins over the `decompose=` callable) and its metadata intent
  (`constraints={"dates": ..., "sources": ..., "entities": ...}` — AND across keys, OR
  within a key, falling back to OR rather than silencing a read). Constraints are a
  **feeder only**: they bank repair candidates and never touch pool scoring or serving
  (pinned).
- **Ingest metadata closes the v0.6 attribution gap.** `retriever.add(..., meta={"title":
  ..., "source": ..., "date": ...})` puts the metadata ON the unit (`source_meta`), and
  the default pool header renders the measured `[title | source | date]` golden path
  without the `pool_header` callable. Meta-less ingests are byte-identical to 0.6.
- **Deprecation status:** the pool-path default flip pre-registered in 0.6.0 is **not
  taken** in 0.7.0 (three of the five gates were not run), so `read_path="unit"` remains
  the default, `serve="pool"` survives unchanged (its removal clock now counts from
  whichever release takes the flip), and no unit-path knob warns yet.

## What's new in v0.6

The pool-first release. Every n=605 number below comes from one frozen rig — a 609-article news corpus, 605 held-out questions, gpt-4.1-mini answerer, strict grading — the same rig the v0.5 numbers were measured on. Full details in the [CHANGELOG](CHANGELOG.md) and [UPGRADE-0.5-to-0.6.md](UPGRADE-0.5-to-0.6.md).

- **`read_path="pool"` — the claim-pool-first read path (opt-in).** Reads are answered by budget-packing the global fresh-claim pool; units remain the ownership / freshness / provenance skeleton. Measured: **0.731 strict accuracy @ 981 mean context tokens** — matching naive top-9 (0.711 @ 1,311) at **~25% fewer tokens** and naive's best measured point (top-12: 0.731 @ 1,729) at **~43% fewer**. The claim is **parity at fewer tokens** (CIs overlap) — not an accuracy beat. Gold-claim serving rank: **p50/p75/p90 = 1/6/15** in pool order.
- **Attribution by default, and a measured header ladder.** `pool_header=None` now renders a built-in per-source header. Same store, same queries: opaque id **0.641** → shipping default **0.678** → your `[title | source | date]` metadata callable **0.731**. The gap is a *unit-metadata* limit (outlet/date live in your corpus, not on the unit) — wire the callable (quickstart above).
- **The behavioral stack (all default-OFF): spans → fallback → repair → keys.** `residual_spans=True` captures fact-bearing sentences the extractor missed as tier-2 spans on the unit (never in the pool). When *your* answerer refuses, `report_refusal(read_id)` returns an attributed retry payload; `report_success(read_id)` confirms the rescue and — with `query_keys=True` — earns a durable alternate key; lossy-marked units self-repair **append-only** on their next rebuild. Measured, driven through the full loop: refusals **91 → 61 (−33%)**, **+3.1 pts** final accuracy at **+2.1% tokens** (same-store comparison), **zero newly-wrong answers**; keyed-class first-pass **0% → 61%** on paraphrase revisits.
- **Adaptive serve gate** (`serve_gate=None`) — adapts against the pool's own noise ceiling; an explicit float disables adaptation (reproducible benches). Shipped only after a $0 replay gate: 605/605 identical serve decisions on both arms, zero builds, zero LLM calls.
- **Hardening & plumbing:** every payload surface carries source attribution (pool payload and escalation raw); cross-owner near-duplicate claims are kept as corroboration; 14 new observability events (`pool_served`, `residual_fallback`, `key_confirmed`, ...); `reranker` hook (serving order only — it can never cause a false serve); `claim_index` BYO pool storage; a v0.5 pool-preview stale-serve hole is fixed.
- **Deprecated:** `serve="pool"` (the v0.5 preview) — still works verbatim in 0.6, removed in v0.7; migrate to `read_path="pool"`. The default read path stays `"unit"` (exact v0.5 behavior); flipping the default is a v0.7 decision behind five pre-registered gates, of which only one (the replay gate) has passed.

### Honest limits (measured, not hypothetical)

- **The refusal fallback flips ~20% of natural refusals** (33% when the payload contains the answer verbatim) — the 68% figure from lab questions holds only where the payload contains the answer by construction. It is a net over the extraction tail, not a second retriever.
- **Query keys can collide across sibling articles** in dense same-topic corpora (observed 3/605 reads; answers still correct). Raise `key_floor` above its 0.85 default there. Keys convert refusal round-trips into first-pass answers; they do **not** raise final accuracy on diverse rewordings.
- **The shipping default header can only attribute what the unit knows** — the 0.678 → 0.731 gap is a unit-metadata limit, closed today by the `pool_header` callable; an optional ingest-time metadata field is a v0.7 item.

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
  superseded by `read_path="pool"` in v0.6 — the preview still works in 0.6, removed in v0.7).
  Held-out n=605: 0.699 accuracy vs 0.579 for unit serving (z=6.66); statistically ties naive's
  k9 arm — its best measured at the time — at 0.79× its tokens; 95% null honesty.
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

This is the **default unit read path** (`read_path="unit"`, exact v0.5 behavior). The opt-in v0.6 [pool path](#whats-new-in-v06) replaces unit routing with global claim-pool packing and makes these unit-routing knobs inert; its own knobs are in [UPGRADE-0.5-to-0.6.md](UPGRADE-0.5-to-0.6.md).

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

For the full standalone MCP server (freshness loop, seven tools, HTTP transport), see the
[next section](#use-it-from-claude-code--cursor-mcp); for LangChain, see
[langchain-coalent](#langchain).

## Benchmark

### Real-world: the pool read path (v0.6, n=605)

Same rig as the v0.5 numbers below — 609 real news articles, 605 frozen held-out questions,
gpt-4.1-mini answerer, strict grading — with naive's own token-scaling curve as the fairness
control, now extended to its best measured point:

| Arm | Accuracy | Context tokens |
|---|:---:|:---:|
| naive top-9 | 0.711 | 1,311 |
| naive top-12 (best measured) | 0.731 | 1,729 |
| **Coalent `read_path="pool"` (defaults + metadata header)** | **0.731** | **981** |

- **The claim is parity at fewer tokens — not an accuracy beat.** CIs overlap on every pair.
  0.731 @ 981 matches naive top-9 accuracy at **~25% fewer** context tokens and naive's best
  measured point at **~43% fewer** (57% of its budget) — plus what retrieval alone cannot do
  (freshness, provenance, behavioral compounding).
- **Serving ranks:** the gold claim sits at **p50/p75/p90 = 1/6/15** in pool order (over
  claim-present queries), with the default cosine ranking — no reranker.
- **Headers are measured, not vibes:** opaque id 0.641 → shipping default 0.678 → your
  `[title | source | date]` callable **0.731**. The 0.731 row above uses the metadata callable;
  wire `pool_header` (see [quickstart](#quickstart)).
- **Behavioral stack** (opt-in): final accuracy **+3.1 pts** at **+2.1% tokens**, refusals
  **−33%**, zero newly-wrong answers — a same-store, same-population comparison driven through
  the full report_refusal/report_success loop.
- The v0.5 anchor on this rig was 0.699 @ ~1,036: the v0.6 rewrite holds the point (CIs
  overlap) with the stale-serve hole fixed and attribution on by default.

### Real-world: MultiHopRAG (v0.5, pre-registered)

609 real news articles, third-party gold questions, answered by gpt-4.1-mini with exact-match
grading — the corpus **maximally friendly to chunk retrieval** (questions are generated from
article sentences), chosen as the adversarial test. We run the fairness control most benchmarks
skip: **naive's own token-scaling curve** on the same stream (k4 0.58 @ 590 tok · k6 0.64 @ 882 ·
k9 0.71 @ 1311, n=605 held-out).

- **Pool serving (`serve="pool"`, warmed cache): 0.699 @ ~1,036 tokens** — beats naive k6
  (paired McNemar z=3.22) and **statistically ties naive's k9 arm — its best measured at the
  time — at 0.79× its tokens** (z=0.60). We do not claim to beat the curve here; the claim is
  match-at-fewer-tokens plus what retrieval alone cannot do (freshness, provenance, compounding
  reuse).
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

One CI reality to know: the project pins mypy's analysis target to 3.10
(`python_version` in `pyproject.toml`), but if numpy >= 2.5 is installed (what the
`[fast]` extra resolves to on Python 3.12+), its stubs use syntax a 3.10 analysis
target cannot parse. In that case run `mypy src --python-version 3.12` — matching
your interpreter — exactly as the CI matrix does.

## Status &amp; license

**Alpha** — the API may change before 1.0. Fully typed (`mypy --strict`), linted, and tested.

Licensed under [Apache-2.0](./LICENSE).

<p align="center"><sub>Context that's trustworthy, not just cheap.</sub></p>
