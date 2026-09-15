# Changelog

All notable changes to this project are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

## [0.7.0]

v0.7 is the self-healing release. The read now ships its own doubt — an opt-in gap
detector that says, per probe, "the source has this fact but my claims don't" — and the
cache gains an explicit repair loop your agent drives: `repair(read_id)` re-extracts what
a build missed and appends it **permanently** (every future read benefits; the cost is
paid once), `reprobe(read_id)` re-ranks with entity probes when the answer is in the pool
but buried, and `serve_unserved(read_id)` force-packs admitted-but-unserved claims after
a refusal. Two upstream parameters (`subs=`, `constraints=`) let a planner hand the read
its decomposition and its metadata intent instead of the cache guessing.

**Everything below is default-OFF and byte-inert until armed** — a 0.6 user who upgrades
and touches nothing gets 0.6 behavior, byte for byte (pinned by dedicated inertness
tests on every knob). The default read path does **not** flip in 0.7 (see the flip-gate
status at the bottom of this entry).

Benchmark rig for the numbers below: the same frozen rig as every anchor since v0.5 —
a 609-article news corpus, 605 held-out questions, gpt-4.1-mini answerer, strict grading,
plus the locked v0.7b adjudication rules (dual-reported; adjudication can only flag
grader-blind string artifacts to a judge, never auto-accept).

<!-- SANCTIONED 2026-09-15 -->
- **Headline**: the full v0.7 composition (subs + gap detector + constraints feeding
  refusal-gated repair, then reprobe, then serve_unserved — each rung firing only on
  failure) measured **0.826 adjudicated (0.825 harness) vs 0.774 for the v0.6
  shipped-max baseline** on the same rig at an identical ~983-token serving budget:
  **+5.3 points with zero extra serving tokens**. <!-- SANCTIONED 2026-09-15 -->
- Refusals fell **69%** (61 → 19). <!-- SANCTIONED 2026-09-15 -->
- The chain also works without caller-supplied decomposition — both compositions
  ship, `subs=` stays caller-owned and is never guessed.
- Cost shape: gating repair on a failed read (the recommended composition) matched
  always-on accuracy at **14% of the extraction calls** — the chain adds cost only on
  reads that failed. <!-- SANCTIONED 2026-09-15 (latency/cost note) -->

### Added

- **The agentic `Result` surface** — the read carries its own provenance and doubt so an
  evaluator node can act without drilling:
  - `Result.sources` — the artifact ids actually behind the served payload (served order
    first, de-duplicated), on both read paths.
  - `Result.max_source_age_s` — the serve's freshness age: MAX served-owner age in
    seconds (the per-read form of the `stats()` aggregate).
  - `Result.probes` / `Result.probe_coverage` — the probe texts this read scored (raw
    query first) and, with the gap detector armed, per-probe
    `{probe, best_claim, best_span, margin, fired}`.
  - `Result.gaps` — the actionable subset: `{probe, span, unit_id, source, kind}` where
    `kind` is `"extraction_hole"` (the evidence tier HAS it → repair terrain) or
    `"corpus_hole"` (nothing reaches the probe → route to a tool/retrieval node).
  - `Result.parent_read_id` — set on `reprobe()` / `serve_unserved()` results; `""` on
    ordinary reads. (`Result.read_id` existed since 0.6.)
- **`gap_detector=True`** (constructor knob, pool path) — observe-only by construction:
  per probe, if a raw evidence sentence outscores every fresh claim by a fixed margin,
  that sentence is banked as a repair candidate on the read's ledger and reported in
  `Result.gaps`. Serving is byte-identical ON vs OFF (pinned). $0 at serve: the evidence
  sentence tier hydrates lazily per unit, embeds once, and is cached.
- **`get(..., subs=[...])`** — planner-owned decomposition: sub-question strings (or
  `{"q", "hyde"}` dicts) feed the same probe-union path as the `decompose=` callable and
  win over it for that read; `subs=[]` sanctions no-decomposition; malformed entries fail
  open to the raw query.
- **`get(..., constraints={...})`** — `{"dates": [...], "sources": [...],
  "entities": [...]}` (intent detection's natural output) matched against unit ingest
  metadata: dates through ONE ISO canonicalizer (never guessed), sources/entities
  case-insensitive substring. **Feeder only**: matched units' best evidence sentences
  join the read's repair-candidate ledger — constraints never touch pool scoring or
  serving (pinned). With ≥2 derivable keys the match is **AND across keys / OR within a
  key's values**; an empty AND set falls back to OR with a
  `constraints_and_fallback` event, so the conjunction can narrow but never silence a
  read's candidates.
- **`repair(read_id)` + `repair_extractor=` (BYO callable)** — the pump. Consumes the
  read's banked candidates (detector fires + constraints matches, best-first), then
  bridge candidates derived from the served claims. Per candidate: one span-anchored
  extractor call (`(span_text, context_region, existing_claims) -> [claims]` — the
  library still never calls an LLM itself), mechanical near-dup rejection, then
  **append-only commit with per-claim provenance**
  (`understanding["_repair_provenance"]`: claim, span, source, origin, via, ts — a
  wrong-but-novel claim stays evictable by inspection). Returns a `RepairReport`
  (`candidates_seen / extracted / rejected / admitted / claims / units_touched`).
  Capped at 8 admissions per call. The store improvement is permanent and persists
  through the normal store path.
- **Repair admission hygiene** — mechanically rejects (before dedup, counted in
  `RepairReport.rejected`) the two measured span-extraction defect shapes: antecedent-free
  pronoun-subject claims ("He was the richest…" with no proper noun anywhere) and
  truncated claims (mid-sentence starts, dangling connectors, trailing punctuation cuts).
- **`reprobe(read_id, hint=None)`** — the mechanical second pass for buried answers:
  harvests proper-noun entities from the read's SERVED claims, pairs them with the
  read's sub-question tails as new probes, one batched embed, then a MAX-union re-rank
  over the unchanged pool (originals included — nothing served can score worse).
  Fresh `Result` with `parent_read_id` set; embeds only, no LLM, no retrieval, no store
  mutation. Self-skips (returns `None`) when there is nothing to harvest.
- **`serve_unserved(read_id)`** — the post-repair refusal rung: a fresh `Result` that
  force-packs, at the head, (a) the question's admitted-but-unserved repaired claims and
  (b) the best claims of the strongest constraint-matched unit absent from the served
  payload, then refills with the original served claims. One embed, no LLM, store never
  mutated. By app contract, call it only on a refusal — a refusal is never a correct
  answer, so this rung cannot break one.
- **Ingest metadata** — `Chunk.meta` (recognized keys `title` / `source` / `date`,
  extras preserved), captured onto the unit as `source_meta` at build, serialized
  round-trip, and rendered by the **metadata-first default pool header**
  (`[title | source | date]` when present). This closes 0.6.0's documented known-limit:
  the measured 0.68-vs-0.73 attribution gap was a unit-metadata limit, and units can now
  carry the metadata. <!-- already-sanctioned v0.6.0 ladder numbers, unchanged -->
  Meta-less ingests keep emitting byte-identical pre-0.7 serde JSON.
- **`decompose=` (constructor callable, default OFF)** — first-pass query decomposition
  for naked deployments: a BYO `callable(query) -> [{"q", "hyde"}, ...]` whose
  sub-questions join the probe union (clamped at 4); an explicit `subs=` always wins.
  The library never calls an LLM itself.
- **New events**: `gap_detector`, `constraints_matched`, `constraints_and_fallback`,
  `repair_applied`, `reprobe`, `reprobe_skipped`, `serve_unserved`,
  `serve_unserved_skipped`, `pool_decomposed`.
- **MCP folder mode** now auto-wires ingest metadata for watched files (title = first
  markdown H1 else filename, source = relative path, date = mtime), so its attribution
  header upgrades to the measured `[title | path | date]` rung automatically.
- **Boundary surfaces**: the MCP server's `get_context` result and the
  `langchain-coalent` retriever's document metadata now carry the read's
  `coverage`, `needs_retrieval`, and `gaps` — additive keys, so an agent framework
  on either integration can drive the failure chain without touching the Python API.

### Changed

- The default pool header is **metadata-first**: units whose ingests carried
  `Chunk.meta` render `[title | source | date]`; meta-less units keep the 0.6 unit-title
  fallback, and the construction warning now names the whole ladder. No behavior change
  for corpora ingested without metadata.

### Fixed

- `OpenAIEmbedder.embed_many` now chunks requests at the provider's hard
  2048-inputs-per-request limit (order preserved, minimum requests). An oversized batch
  previously 400-failed and could silently degrade a feeder that expected one batched
  call (pinned: 5000 texts → [2048, 2048, 904]).

### v0.7 flip gates (pre-registered in 0.6.0) — status

The 0.6.0 entry pre-registered five gates for flipping the default from
`read_path="unit"` to `read_path="pool"` in v0.7. **The flip is not taken in 0.7.0**:
gates (a) structured-template suite, (c) churn soak, and (d) the null suite were not
run as pre-registered, so the default read path is unchanged and the pool path stays
opt-in. Accordingly the deprecation clock has not started: `serve="pool"` (the v0.5
preview) survives unchanged in 0.7 instead of being removed, the unit-path read knobs
carry no `DeprecationWarning` yet, and the 3-month minimum-notice floor now counts from
whichever future release takes the flip.

## [0.6.2]

Metadata-only patch: the MCP registry ownership marker in the README now matches the
published registry namespace (`io.github.nisarg-pujara-vectorlink/coalent`). No code changes.

## [0.6.1]

The distribution release: the 0.6.0 engine, now reachable from Claude Code / Cursor / any
MCP client (the `coalent-mcp` server) and from any LangChain stack (the new
`langchain-coalent` package).

**Upgrading from 0.6.0 is additive-only.** No existing API, default, knob, or store format
changes in any way: the release adds one module (`coalent.mcp`), one console script
(`coalent-mcp`), one public method (`SemanticCache.has_source`), and re-pins the `mcp`
extra to the current MCP SDK. A 0.6.0 user who upgrades and touches nothing gets 0.6.0
behavior, byte for byte. No separate upgrade doc is needed — this paragraph is it.

MCP-specific numbers below come from a 100-question validation run drawn from the same
frozen rig as every 0.6.0 number (the 609-article news corpus, 605 held-out questions,
gpt-4.1-mini answerer, strict grading).

### Known limits (read this first)

- **Folder mode trades accuracy for zero config — measured.** On the same 100 questions,
  the zero-config `--watch` deployment scored **0.46** vs **0.71** for a factory-built
  cache over a purpose-built vector retriever (40 vs 18 refusals). The gap is the cost of
  the generic paragraph chunker plus on-demand keyhole builds, not a cache defect — and it
  is why the bring-your-own-cache factory mode is the primary, documented-first mode.
  Folder mode is the demo wedge.
- **A just-added file is not instantly servable from a warm pool.** The serve gate can
  honestly refuse a question about a brand-new source until a read triggers a build for
  it (the pool has no claims from that file yet). Nothing stale is ever served — the gap
  shows up as a refusal, never a wrong answer. `refresh()` plus a first query warms it.
- **One writer per store (stdio).** Every stdio launch is its own process; pointing two
  MCP client apps at the same `--store` path means two processes writing one SQLite file,
  which is unsupported. For a shared cache across multiple agents or apps, run one
  `--transport http` server — one long-lived process, one cache — which is the validated
  configuration (see below).
- **`--watch` alongside `--cache-factory` only invalidates — it never ingests.** The
  folder scan fires `source_changed` for edited files, but your factory's retriever is
  the only index; the events only line up when your `artifact_id`s equal the
  watch-relative file paths.

### Added

- **The MCP server** — `coalent-mcp`, installed with `pip install "coalent[mcp]"` (add
  `,openai` for folder mode). Serves fresh, attributed facts from a
  provenance-invalidated cache to any MCP client, over stdio (default) or streamable
  HTTP. Two deployment modes, one tool surface:
  - **Bring-your-own-cache factory mode (`--cache-factory module:function`) — the
    primary mode.** Your factory function returns a fully user-constructed
    `SemanticCache`: your vector DB / retriever, your embedder, your LLM, every knob —
    including `store=` for persistence and `pool_header=` for attribution. The server
    adds protocol glue only, and that glue is measured to add **zero quality loss**: over
    a real vector retriever, factory mode reproduced the library's own benchmark result
    **byte-identically** — 0.710 accuracy on the 100-question validation run, identical
    confidence intervals, 100/100 serves, zero errors, 98/100 answer payloads byte-equal
    to the library run. Freshness is signal-driven: your ingestion pipeline calls the
    `source_changed` tool. No `OPENAI_API_KEY` is demanded — your factory brings its own
    models. Folder-mode flags (`--store`, `--budget`) are rejected loudly rather than
    silently ignored.
  - **Folder mode (`--watch DIR`) — the zero-config demo wedge.** Mounts the recommended
    v0.6 deployment (`read_path="pool"`, residual spans, query keys, SQLite persistence)
    over a directory of documents. Every `get_context` call rescans the watched files
    (mtime + content hash) before serving, so you cannot get a stale answer after saving
    a file; attribution ships the measured golden path automatically (`pool_header` =
    `[path | modified YYYY-MM-DD]` — files have metadata, so the bare-header gap never
    opens). An untouched folder restarts fully warm (persisted scan table). Requires
    `OPENAI_API_KEY` and fails loudly without it — never degrading to the lexical
    embedder. Its accuracy cost versus a factory cache is measured and documented above.
  - **Freshness, verified at the answer level.** In validation, editing a watched source
    flipped the very next read's answer (e.g. a cached top-speed fact served 143.7 mph
    before the edit and 178.2 mph immediately after, with `staleness_prevented`
    incrementing). Zero stale serves were observed across all validation runs.
  - **HTTP transport (`--transport http`)** — the SDK's streamable HTTP: one long-lived
    process, many concurrent agents, ONE shared cache (shared compounding, no store
    races; a single lock serializes tool bodies, so two concurrent identical misses build
    once). Validated: two concurrent clients over one shared cache matched the sequential
    reference on all 20 reads — contexts and answers — with zero duplicate builds.
    Optional bearer auth via the `COALENT_MCP_TOKEN` env var (a shared secret for
    localhost / trusted networks; multi-tenant auth is deliberately out of scope).
  - **Seven tools:** `get_context(query, budget?)` (the attributed, budget-packed payload
    + a `read_id` handle), `report_refusal(read_id)` / `report_success(read_id)` (the
    behavioral loop over MCP), `source_changed(artifact_id, text?)` (the BYO freshness
    feed — unchanged content is hash-detected and skipped), `list_sources()`,
    `cache_stats()`, `refresh()`.
- **`SemanticCache.has_source(artifact_id) -> bool`** — true when any cached unit's
  provenance depends on that artifact. The cheap pre-check for change-feed adapters (the
  MCP server uses it to fire `source_changed` only for files some unit actually read);
  additive public API.
- **`langchain-coalent` 0.1.0** — a separate package
  ([integrations/langchain-coalent](integrations/langchain-coalent)) making Coalent a
  LangChain-native freshness/reuse layer, BYO-first: your existing LangChain VectorStore
  (or retriever), `Embeddings`, and chat model become the cache's substrate unchanged.
  `create_coalent_cache(vectorstore, llm=..., embeddings=...)` is the one-call entry;
  `CoalentRetriever` is a drop-in `BaseRetriever` whose documents carry `read_id`,
  `sources`, and `cache_hit` metadata; the refusal→repair loop ships as a runnable
  LangGraph-shaped example. Depends only on `coalent>=0.6` + `langchain-core>=0.3`.

### Changed

- The `mcp` extra now pins `mcp>=2.0` — the server targets the current MCP SDK API.
  (The extra previously existed as a placeholder; nothing imported it.)

## [0.6.0]

v0.6 adds the claim-pool-first read path (`read_path="pool"`) and a default-OFF behavioral
stack (residual spans, refusal fallback, append-only repair, query keys). The default read
path is unchanged: a v0.5 user who upgrades and touches nothing gets v0.5 behavior, modulo
the bugfixes listed under Fixed. The pool path is opt-in in 0.6; the default flip is a
v0.7 decision gated by the pre-registered criteria at the bottom of this entry.

Benchmark rig for every n=605 number below: a 609-article news corpus with 605 frozen
held-out questions, gpt-4.1-mini answerer, strict grading (normalized gold containment) —
the same rig every anchor since v0.5 was measured on.

### Known limits (read this first)

These are measured properties of the release, not edge cases:

- **The pool payload can only attribute what the unit knows.** The shipping default header
  is built from the unit's build query (`## {query[:60]}`). It measured 0.6777 strict
  accuracy on the n=605 news benchmark versus 0.7306 for a caller-supplied metadata header
  (`[title | source | date]`). The gap is a **unit-metadata limit, not a header-format
  problem**: outlet and date live only in corpus metadata, nowhere on the `Cognition` unit,
  so per-outlet questions the payload cannot attribute are refused — correctly. Wire
  `pool_header` to your own metadata (recipe in UPGRADE-0.5-to-0.6.md) to recover the
  difference. An optional ingest-time metadata field is a v0.7 item.
- **The refusal fallback flips fewer refusals on natural tails than on lab questions.**
  On the benchmark's 91 natural first-pass refusals, 18 flipped to strict-correct (~20%
  unconditional); restricted to the 27 refusals whose fallback payload actually contained
  the gold string, 9 flipped (33%). The 68% figure from the hardening lab holds only for
  span-derived questions where the payload always contains the answer.
- **Query keys can collide across sibling articles in dense same-topic corpora.** Observed
  3 reads in 605 where one article's key fired on sibling-article queries (answers were
  still correct in all 3). If your corpus is many near-identical articles on one topic,
  consider raising `key_floor` above its 0.85 default.
- **Keys do not raise final accuracy on diverse rewordings.** Their measured value there
  is converting refusal round-trips into first-pass answers (first-pass 7%→33%, fallback
  retries −70%, final accuracy unchanged in a same-population 3-arm cell). On mild
  paraphrases they fire 90% and give 70% first-pass.

### Added

- **`read_path="pool"` — the claim-pool-first read path (opt-in, default-capable).**
  Every read is answered by budget-packing the global fresh-claim pool; units remain the
  ownership / freshness / build / provenance skeleton. Validated at n=605 on the frozen
  news benchmark: **0.7306 strict accuracy @ 981 mean tokens** (v0.5 anchor 0.699@1036,
  CIs overlap — the point holds), gold-rank p50/p75/p90 = 1/6/15 in pool order over
  claim-present queries, zero regressions on every printed column across the default legs.
  "Default-capable" means: the shipping default constructor passed the full replay gate and
  the n=605 benchmark with no knob tuning; it still ships opt-in per the deprecation
  timeline.
- **Adaptive serve gate (`serve_gate=None`).** An explicit float is absolute (adaptation
  disabled — the reproducible-bench off-ramp); `None` adapts against the pool's own
  null-shaped noise ceiling, clamped at `cov_default + 0.27`. Shipped only after the
  replay gate passed: warmed 609-unit store + 605 frozen queries through the shipping
  default constructor, **605/605 serve decisions on both arms**, zero builds, zero LLM
  spend (enforced by a poisoned synthesizer), packed-payload gold presence within noise of
  the measured 0.699 rig (0.4331 [0.394, 0.473] vs rig 0.4215 [0.383, 0.461]), 981 vs 970
  mean tokens. Exactly 1 of 605 reads sat in the adaptive gate band.
- **Default pool header (attribution by default).** `pool_header=None` no longer serves an
  unattributed payload. The built-in default renders `## {unit.query[:60]}` per source
  group (the build query carries source-identifying text), falling back to
  `[source: {artifact_id}]` for units with no query text. The measured three-point ladder
  on the same frozen store/queries/grader (n=605, strict): opaque id **0.6413** (153
  refusals) → query-title default **0.6777** (136 refusals; **this ships**) → caller
  metadata callable `[title | source | date]` **0.7306** (101 refusals; **documented best
  recipe**). See Known limits for why the remaining gap is a unit-metadata limit.
- **Behavioral stack (all default-OFF; nothing below runs unless switched on):**
  - `residual_spans=True` — build-time sentence audit retains fact-bearing source
    sentences the extractor missed (hard-fact test + max-claim-cosine < `span_tau`,
    capped per unit) as tier-2 spans **on the unit, never in the pool** (zero pool
    crowding). At read time a span outranking every fresh claim by `span_margin` serves
    as a labeled side channel inside the same `serve_budget`.
  - **Refusal fallback** — every read gets a deterministic `Result.read_id` (64-read ring
    buffer). When *your* answerer refuses over a served payload, call
    `report_refusal(read_id)`: the cache re-scores the namespace's residual spans against
    the original query embedding and returns an attributed retry payload (up to 2 spans,
    floor `span_serve_floor=0.35`), or `None`. `report_success(read_id)` is the symmetric
    confirm half. On the n=605 benchmark this machinery delivered a payload on **91/91**
    first-pass refusals and cut net refusals **91 → 61 (−33%)** with zero newly-wrong
    answers.
  - **Append-only repair** — span serves and raw-fallback escalations count toward
    `lossy_threshold` (default 2); a lossy-marked unit repairs on its next rebuild touch
    by *appending* the missing facts (claims are never dropped while the source hash is
    unchanged; replace only on a changed hash). Spans self-retire on recapture.
  - `query_keys=True` (requires `residual_spans` machinery and the pool path) — a
    fallback-rescued read attaches the successful span as a provisional alternate key on
    its unit; `report_success` confirms it durable (serde-persisted). At read time a
    confirmed key scores `max(content_sim, key_sim)` and counts **only at/above
    `key_floor=0.85`**. Compounding probe on the same benchmark: keyed-class first-pass
    accuracy **0% → 61%** on mild paraphrases (11/18), controls untouched (fired 60% vs
    not-fired 58% correct, n=100).
- **Serving hardening (measured findings, all shipped):**
  - **Default attribution on every payload surface**: the pool payload (default header
    above) and the escalation raw floor both carry source attribution; an unattributed
    payload made per-source questions unanswerable by construction.
  - **Within-owner-only near-dup collapse**: stage-1 dedup collapses a unit's *own*
    rephrasings only. Cross-owner near-duplicates — even exact text — are corroboration,
    and per-source questions need the owner's own attributed copy, so they all survive.
  - **Attributed escalation raw**: raw chunks served by the coverage floor are prefixed
    `[source: {artifact_id}]` in the same format as the pool header.
- **New events:** `pool_gate`, `pool_served`, `pool_masked_stale`, `pool_budget_overrun`,
  `rerank_failed`, `rebuild_triggered_by_read`, `build_triggered_by_gap`,
  `unit_marked_lossy`, `unit_repaired`, `residual_served`, `residual_fallback`,
  `key_attached`, `key_confirmed`, `key_fired`. Retained unchanged: `source_changed`,
  `unit_built`, `unit_rebuilt`, `stale_read_prevented`, `admission_reuse`,
  `widen_unavailable`.
- **`Result.read_id`** — deterministic per-read id; the handle for `report_refusal` /
  `report_success`.
- **Constructor guards:** `query_keys=True` with `read_path != "pool"` raises
  `ValueError("query_keys requires read_path='pool'")` at construction. On the unit path,
  keys could attach and confirm yet structurally never fire — a silent-failure class we
  fail loud on instead. Likewise `read_path="pool"` with a `HashingEmbedder` raises at
  construction (claim cosine collapses to keyword overlap; the error names the escape
  hatches).
- **`reranker` hook** (`Callable[[str, list[str]], list[float]] | None = None`) — serving
  order only; the serve/build/floor decisions always read the pre-rerank cosine, so a bad
  reranker can degrade order but never cause a false serve, a skipped build, or a broken
  null refusal. `None` is the default for all of 0.x; no preset sets it.
- **`claim_index` protocol** — BYO pool storage (`ClaimIndex`; built-in numpy/pure-python
  implementations, per-namespace factory form supported).

### Changed

- **`serve_budget` default split:** `None` → 600 on the unit path (v0.5 preserved) and
  1000 on the pool path (reproduces the measured ~1036-token operating point under strict
  header-counting packing). Explicit values always win; `<= 0` raises.
- **Result contract on the pool path** (unit path untouched):
  `understanding = {"claims": [...]}` with no `summary` key; `unit_id` = owner of the
  top-ranked served claim; `confidence` = pre-decision pool coverage (what the gate read);
  `coverage` = final post-build/post-S2; `evidence = []` on a non-escalated serve
  (citations via `drill(unit_id)`); `recalled = []` (unit-path field, kept one version);
  `pool` = served claims in served order. `cache_hit` on the pool path means zero
  synthesis calls ran this read.
- **`stats()` pool additions:** `pool_serves`, `probe_reads`, `admission_reuses`,
  `retrievals`, `rebuilds_by_read`, `builds_by_gap`, `serve_gate_effective`,
  `pool_noise_ceiling`, `pool_claims_fresh`, `pool_claims_total`, `avg_units_per_serve`,
  `pool_scan_slow` — gate miscalibration is a dashboard number, not a silent
  duplicate-build tax.

### Fixed

- **v0.5 pool-preview stale-serve hole.** The preview's `(len, xor-hash)` pool marker
  could return to its prior value after an in-place rebuild within one read (dirty →
  rebuild → fresh, same id, same claim count), serving the old claim texts. Replaced by
  two monotone counters (`_rows_epoch` bumped on any row change, `_status_gen` on any
  freshness flip); a same-size rebuild can no longer restore a prior marker value.

### Deprecated

- `serve="pool"` (the v0.5 experimental preview): preserved verbatim in 0.6, never
  auto-mapped to `read_path="pool"`; removal scheduled v0.7. The 12 unit-path read knobs
  (`hit_threshold`, `hit_margin`, `route_by_claim`, `recall_*`, `select_floor`, ...) are
  inert on the pool path and keep full function on the default path; DeprecationWarning
  v0.7, removal v0.8, minimum-notice floor 3 months after the v0.7 flip ships.

### v0.7 flip gates (pre-registered)

The default flips from `read_path="unit"` to `read_path="pool"` in v0.7 ONLY if all hold:

> (a) the structured template suite: pool ≥ unit at equal budgets; (b) the FRAMES
> benchmark complete with pool ≥ token-matched naive; (c) churn soak green; (d) the
> 100-question unanswerable (null) suite ≥93% refusal under the adaptive gate; (e) the
> replay gate holds on the shipping default constructor.

Status at 0.6.0: (e) passed (605/605 both arms, this release); (a)–(d) are open v0.7 work
and are not claimed here.

## [0.5.1]

Namespace-isolation fixes for two v0.5 features, found by the v0.6 design review's
adversarial pass and confirmed in code:

### Fixed
- `serve="pool"` now filters the claim pool by namespace — one namespace's claims can no
  longer be served into another namespace's context (isolation bug in the experimental
  pool preview; single-namespace users were unaffected).
- `provenance_admission` containment is now namespace-scoped — a unit in a foreign
  namespace can no longer suppress a legitimate build in yours.

Both pinned by new tests (`test_pool_is_namespace_isolated`,
`test_containment_ns_scoped_both_modes`).

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
