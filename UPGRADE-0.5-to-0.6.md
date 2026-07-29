# Upgrading from 0.5 to 0.6

> Numbers below come from a 605-question benchmark over a 609-article news corpus
> (gpt-4.1-mini answerer, strict grading) plus a $0 replay gate; the full evidence map is
> in [CHANGELOG.md](CHANGELOG.md).

## 1. TL;DR — nothing changes unless you opt in

The default read path (`read_path="unit"`) is v0.5 behavior, byte-for-byte, pinned by the
paired-replay contract suite — modulo one listed bugfix (the v0.5 pool-*preview* stale-serve
marker hole; if you never set `serve="pool"`, it never affected you). Your stores load
without migration; v0.6 writes the legacy store format by default, so a rollback to v0.5
reads your store unchanged.

Two constructor calls now fail loudly instead of silently misbehaving:

- `read_path="pool"` without a semantic embedder (i.e. under the lexical `HashingEmbedder`)
  raises `ValueError` — set `OPENAI_API_KEY` or pass `embedder=`.
- `query_keys=True` without `read_path="pool"` raises
  `ValueError("query_keys requires read_path='pool'")` — on the unit path keys would
  attach and confirm but structurally never fire.

If you have a keyless v0.5 smoke test that constructs a pool-preview cache, it may now
need an embedder to construct.

## 2. Opting into the pool read path

```python
from coalent import SemanticCache

cache = SemanticCache(retriever, synthesizer, read_path="pool")
r = cache.get("what did the report say about Q3 revenue?")
r.context["pool"]     # the packed, attributed claim payload — hand this to your answerer
r.pool                # served claims in served order (owner + score per claim)
```

`serve_budget` is the one knob most users touch: `None` resolves to **1000** on the pool
path (the measured operating point: 0.7306 strict accuracy @ 981 mean tokens, n=605) and
**600** on the unit path (v0.5 preserved). If you tuned the v0.5 preview, set
`serve_budget` to the **measured payload tokens** from your `pool_served` logs, not the
nominal number — the v0.5 preview under-counted headers.

Behavior deltas to check when you flip (all pool-path only):

| What | v0.5 unit path | v0.6 pool path |
|---|---|---|
| `understanding` | `{"summary": ..., "claims": ...}` | `{"claims": [...]}` — **no `summary` key** |
| `unit_id` | the matched unit | owner of the **top-ranked served claim** |
| `confidence` | blended unit score | pre-decision pool coverage (re-tune any threshold you compare it against) |
| `evidence` | populated on hits | `[]` on non-escalated serves — cite via `drill(unit_id)` |
| `cache_hit` | match above threshold | zero synthesis calls this read (your hit-rate series discontinues here; watch `stats()["probe_reads"]`) |
| `recalled` | populated | `[]` (field kept one version) |
| unit read knobs | active | inert (`hit_threshold`, `hit_margin`, `route_by_claim`, `recall_*`, `select_floor`, ...) |

## 3. The `pool_header` golden path

The pool payload attributes every source group with a header. Three measured rungs
(same frozen store, 605 queries, strict grader):

| Header | Accuracy | Refusals |
|---|---|---|
| opaque id `[source: art:N]` | 0.6413 | 153 |
| built-in default `## {unit.query[:60]}` (what ships) | 0.6777 | 136 |
| **your metadata callable `[title \| source \| date]` (recommended)** | **0.7306** | **101** |

The gap between the last two is a unit-metadata limit: outlet and date live in *your*
corpus metadata, nowhere on the unit — so wire them in. The recipe (this exact shape is
what the 0.7306 measurement used):

```python
# your own corpus metadata, keyed by artifact id
meta = {art["artifact_id"]: art for art in corpus}

def header(unit) -> str:
    aid = unit.evidence[0].artifact_id if unit.evidence else ""
    art = meta.get(aid)
    if art is None:
        # an empty return serves the group HEADERLESS (the built-in default does NOT
        # take over once a callable is set) — emit your own fallback line instead:
        return f"[source: {aid or unit.id}]"
    return f"[{art['title']} | {art['source']} | {art['published_at'][:10]}]"

cache = SemanticCache(retriever, synthesizer, read_path="pool", pool_header=header)
```

Notes: a raising header callable is swallowed (the group serves headerless — attribution
degrades, serving never breaks); header tokens count against `serve_budget`. If your
questions never ask "which source said X", the default header is fine; the refusals the
default takes on per-outlet questions are honest refusals of questions its payload cannot
attribute.

## 4. The behavioral-stack recipe (opt-in, default-OFF)

The stack: `residual_spans=True` captures fact-bearing sentences the extractor missed as
tier-2 spans on the unit; a refusal from *your* answerer triggers `report_refusal(read_id)`
→ an attributed retry payload; `report_success(read_id)` confirms the rescue, earns a
durable query key (`query_keys=True`), and the unit self-repairs append-only on its next
rebuild touch. Measured on the n=605 benchmark (same store): final accuracy +3.1 pts,
refusals −33%, tokens +2.1%, zero newly-wrong answers; keyed-class first-pass 0% → 61% on
paraphrase revisits.

The cache never sees your answerer, so **you** wire the loop. LangGraph-shaped example
(shape, not a pinned runnable — adapt the state/LLM bits to your graph; a runnable
no-API-key version is [examples/pool_read_path.py](examples/pool_read_path.py)):

```python
cache = SemanticCache(
    retriever, synthesizer,
    read_path="pool",
    residual_spans=True,     # span capture + refusal fallback
    query_keys=True,         # durable keys from confirmed rescues (needs pool path)
)

def retrieve(state):
    r = cache.get(state["question"])
    return {"read_id": r.read_id, "payload": r.context["pool"], "retried": False}

def synthesize(state):
    answer = llm.invoke(PROMPT.format(context=state["payload"], q=state["question"]))
    return {"answer": answer}

def route_after_answer(state):
    if is_refusal(state["answer"]) and not state["retried"]:   # your own refusal detector
        return "repair"
    return "confirm"

def repair(state):
    retry = cache.report_refusal(state["read_id"])   # None when no span clears the floor
    if retry is None:
        return {"retried": True}                     # accept the refusal
    return {"payload": state["payload"] + "\n\n" + retry, "retried": True}

def confirm(state):
    # confirm ONLY a retry that actually succeeded — this is what earns the durable key
    if state["retried"] and not is_refusal(state["answer"]):
        cache.report_success(state["read_id"])
    return {}

g = StateGraph(State)
g.add_node("retrieve", retrieve); g.add_node("synthesize", synthesize)
g.add_node("repair", repair);     g.add_node("confirm", confirm)
g.set_entry_point("retrieve")
g.add_edge("retrieve", "synthesize")
g.add_conditional_edges("synthesize", route_after_answer,
                        {"repair": "repair", "confirm": "confirm"})
g.add_edge("repair", "synthesize")   # one retry pass, then route_after_answer falls through
g.add_edge("confirm", END)
```

Wiring rules that matter:

- Call `report_refusal` only on genuine refusals, with the same read's `Result.read_id`
  (a 64-read ring buffer; stale ids return `None`, never raise).
- Call `report_success` only when the **retry** answered — it confirms provisional keys
  durable. Unknown/expired ids are a silent no-op.
- Bound your retry loop (`retried` flag above): one fallback round-trip per read is the
  measured regime.
- Expect the fallback to flip roughly 20% of natural refusals unconditionally (33% when
  the payload contains the answer verbatim) — it is a net over the extraction tail, not a
  second retriever. Details and caveats: the "Known limits" section of
  [CHANGELOG.md](CHANGELOG.md).

## 5. Knob table (v0.6 additions; defaults as shipped)

| Knob | Default | Gates |
|---|---|---|
| `read_path` | `"unit"` | Which read path runs. `"unit"` = byte-identical v0.5; `"pool"` = claim-pool-first serving. |
| `serve_budget` | `None` → 600 unit / 1000 pool | Packed payload size (tokens, headers counted). The one knob most users should touch. |
| `serve_gate` | `None` (adaptive) | Pool serve-vs-build decision. Explicit float = absolute (reproducible benches); `None` adapts to the pool's null-shaped noise ceiling, clamped at `cov_default + 0.27`. |
| `pool_header` | `None` → built-in `## {query[:60]}` / `[source: id]` | Per-source-group attribution line. Wire your metadata here (§3). |
| `reranker` | `None` | Serving **order** only; never serve/build/floor decisions. BYO callable. |
| `claim_index` | `None` → built-in | Pool storage (numpy in-proc by default; BYO adapter or per-namespace factory). |
| `coverage_floor` | embedder-derived (0.28 OpenAI-class) | Escalation floor, now measured over the pool: below it, attributed raw chunks are appended. |
| `residual_spans` | `False` | The whole tier-2 machinery: build-time span capture, side-channel serving, `report_refusal` fallback, lossy marking, append-only repair. |
| `span_tau` | `0.62` | Build-time capture: a fact-bearing sentence whose max claim cosine is below this becomes a span. |
| `span_margin` | `0.0` | Read-time anomaly rule: a span must outrank the best fresh claim by this margin to serve. |
| `span_serve_floor` | `0.35` | `report_refusal` retry floor: minimum span-query cosine to return a retry payload. |
| `lossy_threshold` | `2` | Span serves + raw fallbacks against one unit before it is marked lossy (→ append-only repair on next rebuild touch). |
| `query_keys` | `False` | Behavioral alternate keys from confirmed rescues (requires `read_path="pool"`, loud guard). |
| `key_floor` | `0.85` | Minimum key-query cosine for a key to count at all (below = ignored entirely). Raise it in dense same-topic corpora — sibling-article collisions are a measured watch item. |

Unchanged v0.5 knobs keep their positions, defaults, and (on the unit path) their exact
meaning.

## 6. Deprecations, store compatibility, rollback

- **`serve="pool"`** (the v0.5 preview) still works verbatim in v0.6 and is never
  auto-mapped to `read_path="pool"`; it is removed in v0.7. Migrate by replacing
  `serve="pool"` with `read_path="pool"` and re-checking §2's behavior deltas.
- **Unit-path read knobs** (`hit_threshold`, `adaptive_hit`, `hit_margin`,
  `understanding_weight`, `route_by_claim`, `cross_unit_recall`, `recall_threshold`,
  `recall_limit`, `recall_raw`, `recall_bridge`, `select_floor`, `learn_on_escalation`)
  are inert on the pool path; unchanged on the default path. Scheduled: warning v0.7,
  removal v0.8 (no earlier than 3 months after the v0.7 flip ships).
- **Rollback safety:** v0.6 writes the legacy list-of-floats store format by default. A
  cache that never opts into the behavioral stack emits byte-identical v0.5 JSON. The new
  unit fields (`residual_spans`, `query_keys`, `span_hits`, `lossy`) are written only when
  set; a v0.5 reader ignores unknown keys, so a v0.6 store rolls back cleanly either way
  (the earned spans/keys are simply dropped on a v0.5 load). v0.6 has no other store
  write format: a compact binary embedding encoding is deferred (read-compat first,
  write-default no earlier than v0.7).
- **The v0.7 default flip** (pool becomes the default) happens only if the five
  pre-registered gates hold; see [CHANGELOG.md](CHANGELOG.md). Nothing you do in v0.6
  commits you to it: `read_path="unit"` remains available through v0.7.
