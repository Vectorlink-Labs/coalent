# Coalent — Upgrade guide: v0.2.0 → v0.3.0

> **For the AI coding assistant reading this:** your job is to upgrade a project's
> Coalent integration from `coalent==0.2.0` to `coalent==0.3.0`. v0.3 is **additive and
> backward-compatible** — existing code keeps working. So: (1) bump the version, (2) apply
> the **Minimal migration** below (one or two lines, big quality win), (3) optionally adopt
> the enhancements, (4) run the **Test** section. Do not rewrite working call sites you
> don't need to. Every code block is runnable.

---

## TL;DR of what changed (0.2.0 → 0.3.0)

- The cache now **matches by the *meaning* of the cached understanding**, not just the seed query — so `"exchange policy"` stops false-hitting `"leave policy"`, while paraphrases still hit.
- **Coverage + escalation are now semantic** (per-claim cosine). A cache hit that doesn't actually cover the query **falls back to fresh raw** automatically (the "RAG floor"). Two new result fields expose this: `result.coverage` and `result.escalated`.
- A real **embedder** (`OpenAIEmbedder`) gives semantic cache hits out of the box. Without one you get a lexical fallback (`HashingEmbedder`) that matches on keyword overlap + warns.
- New **opt-in knobs**: `coverage_scorer` (cross-encoder/LLM containment check), `coverage_ceiling`, `route_by_claim`, `relevance_gate`, and `depth` on `LLMSynthesizer`.
- `stats()` now reports `reads`, `hits`, `escalations`, `escalation_rate`, `hit_rate`.

**Nothing is removed from the public API.** (Internally the old lexical coverage gate was replaced by semantic coverage — not a public symbol, no action needed.) Persisted caches from 0.2 **auto-backfill** their new embeddings on load.

---

## Step 1 — Install

```bash
pip install -U "coalent[openai]==0.3.0"     # recommended: enables OpenAIEmbedder + provider
# or, core only:
pip install -U "coalent==0.3.0"
```

Verify:

```python
import coalent
assert coalent.__version__ == "0.3.0"
```

---

## Step 2 — Minimal migration (do this — small diff, biggest win)

The single most valuable change is giving the cache a **real embedder** so it matches by meaning. Then read the two new coverage fields.

**BEFORE (v0.2 — lexical keyword matching, paraphrases can miss):**
```python
from coalent import SemanticCache, LLMSynthesizer, OpenAIProvider

cache = SemanticCache(
    retriever,
    LLMSynthesizer(OpenAIProvider(), model="gpt-4o-mini"),
)

result = cache.get("what is our refund policy?")
answer_context = result.context["understanding"]
```

**AFTER (v0.3 — semantic matching + coverage awareness):**
```python
from coalent import SemanticCache, LLMSynthesizer, OpenAIProvider, OpenAIEmbedder

cache = SemanticCache(
    retriever,
    LLMSynthesizer(OpenAIProvider(), model="gpt-4o-mini"),
    embedder=OpenAIEmbedder(),          # <-- match queries by MEANING (recommended)
)

result = cache.get("what is our refund policy?")
answer_context = result.context["understanding"]

# NEW in 0.3 — know how well the cache covered the query, and whether it fell back to raw:
if result.escalated:
    # the cached understanding under-covered this query, so Coalent pulled fresh raw.
    answer_context = result.context          # includes raw chunks under result.context["raw"]
log.info("coverage=%.2f hit=%s escalated=%s", result.coverage, result.cache_hit, result.escalated)
```

> Notes:
> - With `coalent[openai]` installed **and** `OPENAI_API_KEY` set, the cache auto-uses OpenAI embeddings even if you omit `embedder=`. Passing it explicitly is clearer and recommended.
> - Without a real embedder, Coalent uses `HashingEmbedder` (keyword overlap) and prints a warning — fine for demos, **not** for production semantic matching.
> - `cache.get(query)` is unchanged — you don't touch call sites; the matching improves under the hood.

---

## Step 3 — New `SemanticCache` parameters (all keyword-only, all optional)

```python
SemanticCache(
    retriever,
    synthesizer,
    *,
    embedder=None,                       # OpenAIEmbedder() / FunctionEmbedder(fn) / HashingEmbedder()
    hit_threshold=None,                  # auto-derived per embedder when None
    coverage_floor=None,                 # auto-derived per embedder when None (now SEMANTIC)
    understanding_weight=0.7,            # blend: 0.7*topic(query↔understanding) + 0.3*seed(query↔seed-query)
    route_by_claim=False,                # match by the unit's best CLAIM (late interaction)
    enable_coverage_escalation=True,     # the RAG-floor safety net (on by default)
    coverage_scorer=None,                # optional (query, understanding) -> float containment check
    coverage_ceiling=1.0,                # two-tier: consult coverage_scorer only in [floor, ceiling)
    relevance_gate=None,                 # optional (query, chunks) -> chunks reranker/filter
    learn_behavior=True,                 # remember queries that hit a unit
    max_hit_queries=16,
    strategy=ContextStrategy.CONTEXT_FIRST,  # unchanged
    store=None,                          # unchanged (SQLite/Redis/InMemory)
    freshness=None,                      # unchanged (FreshnessPolicy)
)
```

---

## Step 4 — Optional enhancements (adopt only if the app needs them)

**a) Containment-grade coverage with a `coverage_scorer`** — for the "topically adjacent but missing the specific fact" case that cosine can't catch (e.g. a "sick days" query hitting an annual-leave unit). Two-tier via `coverage_ceiling` keeps it cheap (only consulted on borderline queries):
```python
def my_entailment_scorer(query: str, understanding: dict) -> float:
    # return 0.0..1.0 — your cross-encoder / NLI model / one-token LLM yes-no
    ...

cache = SemanticCache(
    retriever, synth, embedder=OpenAIEmbedder(),
    coverage_scorer=my_entailment_scorer,
    coverage_ceiling=0.65,   # use the scorer only when cosine is in [coverage_floor, 0.65)
)
```

**b) `route_by_claim=True`** — for fat multi-claim units; routes a query to the unit holding a claim *about it* instead of the averaged centroid.

**c) `relevance_gate`** — drop irrelevant retrieved chunks before synthesis (bring your own reranker):
```python
cache = SemanticCache(retriever, synth, relevance_gate=lambda q, chunks: my_rerank(q, chunks)[:5])
```

**d) Synthesis `depth`** — trade understanding completeness vs cost:
```python
LLMSynthesizer(OpenAIProvider(), model="gpt-4o-mini", depth=0.8)  # 0.0 terse · 0.5 balanced · 1.0 exhaustive
```

**e) Threshold tuning (optional, advanced)** — `coalent.calibrate_thresholds(...)` (with labeled positives/negatives) or `coalent.suggest_thresholds(...)` (labels-free); `coalent.default_thresholds_for(embedder)` returns the auto defaults. See coalent.ai/docs for exact signatures. Don't hardcode v0.2 thresholds — `coverage_floor` is now a semantic cosine, not a lexical fraction, and differs per embedder.

---

## `Result` object reference (what `cache.get()` returns)

```python
result.cache_hit      # bool — was it a warm semantic hit
result.coverage       # float 0..1 — how well the unit covered THIS query   (NEW in 0.3)
result.escalated      # bool — did it fall back to fresh raw for this query  (NEW in 0.3)
result.context        # {"understanding": <query-relevant slice>, "raw": [<chunks> if escalated/strategy]}
result.understanding  # full understanding dict (summary, claims, entities, facts, ...)
result.evidence       # list[Chunk] — retained raw evidence (the RAG floor)
result.unit_id        # cache unit id
result.related        # list[Related] — cross-unit links (shared entity/source)
result.raw_text       # property: evidence joined as text
```

---

## Step 5 — Test the upgrade

Drop this into a test or a scratch script in the employee app and confirm it passes against `0.3.0`.

```python
import coalent
from coalent import SemanticCache, InMemoryRetriever, StubSynthesizer

# 1) version
assert coalent.__version__ == "0.3.0", coalent.__version__

# 2) smoke test: cold -> warm hit -> source change -> rebuild
retriever = InMemoryRetriever()
retriever.add("hr:leave", "Annual leave: 21 days per year. Carryover up to 5 days.")
cache = SemanticCache(retriever, StubSynthesizer())   # StubSynthesizer = no API key needed

r1 = cache.get("how many annual leave days?")
assert r1.cache_hit is False                          # cold build
r2 = cache.get("how many annual leave days?")
assert r2.cache_hit is True                           # warm hit
assert 0.0 <= r2.coverage <= 1.0                      # NEW field present
assert isinstance(r2.escalated, bool)                 # NEW field present

# 3) provenance invalidation
cache.source_changed("hr:leave", text="Annual leave: now 25 days per year.")
r3 = cache.get("how many annual leave days?")
assert r3.cache_hit is False                          # rebuilt only the unit that used hr:leave

# 4) observability (NEW in 0.3)
s = cache.stats()
assert {"reads", "hits", "escalations", "escalation_rate", "hit_rate"} <= set(s)
print("stats:", s)
```

Then, **with the real embedder** (the production path), run the app's own integration tests and watch:

```python
# after switching to embedder=OpenAIEmbedder():
print(cache.stats()["hit_rate"])          # should be similar or BETTER (semantic matches paraphrases)
print(cache.stats()["escalation_rate"])   # if high, the understanding is too thin -> raise synthesizer depth
```

**Acceptance checklist for the AI:**
- [ ] `coalent.__version__ == "0.3.0"` and app imports unchanged.
- [ ] Cache constructed with `embedder=OpenAIEmbedder()` (prod) — no `HashingEmbedder` warning in logs.
- [ ] App now reads `result.coverage` / `result.escalated` where it makes decisions about answer quality.
- [ ] Existing app tests pass; the smoke test above passes.
- [ ] `cache.stats()["escalation_rate"]` is sane (not ~1.0). If high, raise `LLMSynthesizer(depth=...)`.

---

## Gotchas / what could trip the AI up

1. **Don't hardcode 0.2 thresholds.** `coverage_floor`/`hit_threshold` now auto-derive per embedder and `coverage_floor` is semantic. If the app set explicit numeric thresholds in 0.2, drop them (let them auto-derive) or re-tune for 0.3.
2. **`default_embedder` is internal** — not a public import. Use `OpenAIEmbedder()` / `FunctionEmbedder(fn)` explicitly, or rely on the auto-pick when `coalent[openai]` + `OPENAI_API_KEY` are present.
3. **`OPENAI_API_KEY`** must be set in the app's environment for `OpenAIEmbedder` and `OpenAIProvider`.
4. **`cache.get()` signature is unchanged** — `get(query, *, namespace=None, related=3, strategy=None)`. No call-site rewrites needed for the core upgrade.
5. **Escalation is on by default.** If the app must never pay an extra retrieval on a hit, set `enable_coverage_escalation=False` — but you then lose the RAG floor (the cache can serve a thin answer). Prefer leaving it on.
