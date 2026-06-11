<!-- After you create the GitHub repo, swap the logo src below to the absolute raw URL
     (https://raw.githubusercontent.com/<you>/coalent/main/brand/wordmark.svg) so it
     renders on PyPI too. Relative paths only render on GitHub. -->
<p align="center">
  <img src="brand/wordmark.svg" alt="Coalent" width="300">
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
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> ·
  <a href="#how-it-works">How it works</a> ·
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

- 🧠 **Understanding, not chunks.** It caches the decision-ready understanding your LLM produced — *and retains the raw evidence with every unit*, so it can never return less than plain retrieval.
- ♻️ **Reuse across queries and agents.** A semantic cache keyed by query *meaning*: ask again, or from another agent, and it's a warm hit — no re-retrieval, no re-synthesis.
- 🌿 **Fresh by provenance.** Every unit remembers the exact sources it used. When one changes, only the units that actually used it go stale — precisely, automatically, and lazily.

Coalent sits **above retrieval** — bring any retriever (vector DB, hybrid search, GraphRAG, tools, APIs). It's the freshness-and-reuse layer, not another retriever.

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

Wire in a real model — any text-in / text-out LLM works:

```python
from coalent import SemanticCache, LLMSynthesizer, OpenAIProvider

cache = SemanticCache(retriever, LLMSynthesizer(OpenAIProvider(), model="gpt-4o-mini"))
```

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

The same query + source-change workload across three systems, scored by an independent oracle (reproduce it with `coalent.evaluation.harness.run_benchmark`):

| System | Accuracy | Stale answers | Token cost |
|---|:---:|:---:|:---:|
| Always re-read (naive RAG) | 100% | 0% | 70 |
| Cached, no invalidation | 80% | **20%** | 35 |
| **Coalent** | **100%** | **0%** | **49** |

As correct as re-reading everything, **~30% cheaper**, with **zero stale answers** — and a change to an *unrelated* source causes no staleness, by construction (provenance precision).

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

📚 **Full docs:** `https://YOUR-DOCS-URL` — concepts, provenance & freshness, retrievers, synthesizers, persistence, worked examples (vector search, MCP & tools, agents), and the complete `get()` / data-model reference.

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
