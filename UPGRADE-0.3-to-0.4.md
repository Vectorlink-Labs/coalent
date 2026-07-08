# Coalent — Upgrade guide: v0.3.0 → v0.4.0

> **For the AI coding assistant reading this:** your job is to upgrade a project's
> Coalent integration from `coalent==0.3.0` to `coalent==0.4.0`. v0.4 is **additive**, with
> **two default changes** that are strictly better on structured / reuse-heavy corpora and
> have a one-line escape hatch back to exact v0.3 behaviour. So: (1) bump the version, (2)
> **usually do nothing else** — the cache just keeps more facts and answers multi-hop
> questions now, (3) read the *behaviour change* note so you're not surprised, (4) run the
> **Test** section. Do not rewrite working call sites. Every code block is runnable.

---

## TL;DR of what changed (0.3.0 → 0.4.0)

- **Extractive understanding is now the default.** `LLMSynthesizer` builds a
  **query-independent list of atomic, source-grounded claims** instead of a question-shaped
  prose summary. One cached unit now answers many *different* later questions, and it keeps
  every number (a prose summary silently dropped facts). Escape hatch: `LLMSynthesizer(extract=False)`.
- **Cross-unit claim recall is now the default.** When the single best-matched unit
  under-covers a query, the cache pools per-claim memory across **all** fresh units and
  surfaces the best-matching claims — so it can answer **multi-hop** questions whose evidence
  spans documents, at **zero extra LLM calls**. It is **dormant (free) on single-hop** and
  auto-disabled under a non-semantic embedder. Escape hatch: `SemanticCache(cross_unit_recall=False)`.
- **New opt-in knobs (default OFF):** `hit_margin`, `select_floor`, `residual_floor`.
- **New result field:** `result.recalled` (the cross-unit claims surfaced for this read).
- **Nothing is removed from the public API.** Persisted 0.3 caches load fine (units rebuild
  lazily; new units are extractive).

**The one behaviour change to know:** cached `understanding` is now atomic claims, not prose.
`understanding["claims"]` is the substance; `understanding["summary"]` may be terse or empty
under extraction. If your app specifically renders the prose summary, either read the claims
or pass `extract=False`.

---

## Step 1 — Install

```bash
pip install -U "coalent[openai]==0.4.0"     # recommended: OpenAIEmbedder + provider
# or, core only:
pip install -U "coalent==0.4.0"
```

Verify:

```python
import coalent
assert coalent.__version__ == "0.4.0"
```

---

## Step 2 — Migration (usually: nothing)

If you were already on v0.3 with a real embedder, **you do not have to change any code** — the
same `cache.get(query)` now builds extractive units and can recall across units. You simply
get more complete answers.

**If you want the exact v0.3 read behaviour** (prose understanding, single-unit only):

```python
from coalent import SemanticCache, LLMSynthesizer, OpenAIProvider, OpenAIEmbedder

cache = SemanticCache(
    retriever,
    LLMSynthesizer(OpenAIProvider(), model="gpt-4o-mini", extract=False),  # <- v0.3 prose
    embedder=OpenAIEmbedder(),
    cross_unit_recall=False,                                               # <- v0.3 single-unit
)
```

---

## Step 3 — New parameters (all keyword-only, all optional)

```python
# LLMSynthesizer
LLMSynthesizer(provider, *, extract=True, ...)   # extract now defaults TRUE

# SemanticCache — new v0.4 knobs (defaults shown)
SemanticCache(
    retriever, synth, *,
    cross_unit_recall=True,      # NEW default — multi-hop recall; free on single-hop
    recall_threshold=None,       # fires when coverage < this (None -> inherits coverage_floor)
    recall_limit=6,              # max claims pooled per read
    hit_margin=0.0,              # OPT-IN: refuse a unit that only ties a neighbour by < margin
    select_floor=None,           # OPT-IN: serve atoms by meaning (per-claim cosine >= floor)
    residual_floor=None,         # OPT-IN (build-time): keep number spans the extractor dropped
    residual_limit=24,
    # ...all v0.3 params unchanged (embedder, coverage_floor, coverage_scorer, ...)
)
```

---

## Step 4 — Optional: get the FULL multi-hop win

Cross-unit recall is on by default, but at the default threshold it fires only when a unit
genuinely under-covers a query (so single-hop stays free). For **cross-document / multi-hop**
workloads — "compare plan A vs plan B", "what changed between v1 and v2" — raise the trigger
so recall pools claims on partially-covered reads too:

```python
cache = SemanticCache(
    retriever, synth, embedder=OpenAIEmbedder(),
    cross_unit_recall=True,
    recall_threshold=0.7,        # more eager recall -> full multi-hop bridging
)

r = cache.get("how does the Growth plan's API limit compare to Enterprise?")
for rc in r.recalled:           # claims pooled from OTHER units (the bridge facts)
    print(rc.unit_id, rc.claim, rc.score)
```

---

## Step 5 — Test the upgrade

```python
import coalent
from coalent import SemanticCache, InMemoryRetriever, StubSynthesizer

assert coalent.__version__ == "0.4.0", coalent.__version__

retriever = InMemoryRetriever()
retriever.add("hr:leave", "Annual leave: 21 days per year. Carryover up to 5 days.")
cache = SemanticCache(retriever, StubSynthesizer())   # no API key needed

r1 = cache.get("how many annual leave days?")
assert r1.cache_hit is False                          # cold build
r2 = cache.get("how many annual leave days?")
assert r2.cache_hit is True                           # warm hit
assert isinstance(r2.recalled, list)                  # NEW field present (recalled claims)

# provenance invalidation unchanged
cache.source_changed("hr:leave", text="Annual leave: now 25 days per year.")
assert cache.get("how many annual leave days?").cache_hit is False   # only that unit rebuilt
```

**Acceptance checklist for the AI:**
- [ ] `coalent.__version__ == "0.4.0"` and app imports unchanged.
- [ ] App still passes its own tests with the new extractive defaults (understanding is now
      `claims`, not prose — adjust any code that read `understanding["summary"]` as the answer).
- [ ] For multi-hop features, `recall_threshold` raised (~0.7) where cross-document answers matter.
- [ ] If exact v0.3 output is required somewhere, that call site passes `extract=False` and
      `cross_unit_recall=False`.

---

## Gotchas / what could trip the AI up

1. **Understanding is now claims, not prose.** The biggest change. If the app rendered
   `result.understanding["summary"]` as the answer, switch to `result.context` /
   `understanding["claims"]`, or pass `extract=False` to keep prose.
2. **Recall only ADDS context, never replaces.** `result.recalled` and
   `context["understanding"]["recalled_claims"]` are additive; the matched unit's own claims
   are still served. Recall never fires under a non-semantic embedder (HashingEmbedder).
3. **`hit_margin` / `select_floor` are OPT-IN and situational.** Leave them off unless you
   have a specific need (contradiction-heavy corpus / paraphrase-heavy queries). On clean
   data `hit_margin` costs rebuilds for no accuracy gain.
4. **`cache.get()` signature is unchanged** — no call-site rewrites needed for the upgrade.
