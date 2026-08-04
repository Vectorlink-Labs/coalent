# langchain-coalent

[Coalent](https://pypi.org/project/coalent/) as a LangChain-native freshness/reuse
layer — **BYO-first**: your existing LangChain vector store (or retriever),
embeddings, and chat model become the substrate of a provenance-invalidated
semantic cache. Nothing about how you built them changes.

```bash
pip install langchain-coalent
```

Depends only on `coalent>=0.6` and `langchain-core>=0.3` — no `langchain-community`,
no `langgraph`.

## The four surfaces

### 1. `create_coalent_cache` — one call over your stack

```python
from langchain_coalent import create_coalent_cache

cache = create_coalent_cache(
    my_vectorstore,            # any VectorStore or BaseRetriever — unchanged
    llm=my_chat_model,         # any BaseChatModel, used as YOU configured it
    embeddings=my_embeddings,  # any Embeddings — keys the cache semantically
    # ...every Coalent knob passes through:
    # hit_threshold=..., serve_budget=..., key_floor=..., preset="multi_hop", ...
)
```

Recommended defaults applied (only where you left the knob unset):

- `read_path="pool"` — the v0.6 measured operating point — whenever you supplied a
  semantic embedder (`embeddings=`, or a non-hashing Coalent `embedder=`). Without
  one the factory stays on the unit read path rather than guess.
- `pool_header` — per-source `[artifact_id]` attribution headers on the pool path
  (pass your own for `[title | source | date]` richness).
- Behavioral knobs (`residual_spans`, `query_keys`, ...) stay **opt-in**, exactly
  as in Coalent itself.

Your chat model is invoked **as you configured it** — the synthesizer's
`model`/`max_tokens`/`temperature` are not forwarded (there is no portable kwarg
contract across LangChain chat integrations). Temperature 0 on your model is
recommended for the strict-JSON synthesis contract.

### 2. `CoalentRetriever` — the cache as a LangChain retriever

```python
from langchain_coalent import CoalentRetriever

retriever = CoalentRetriever(cache=cache)          # drop-in BaseRetriever
docs = retriever.invoke("what is our leave policy?")

docs[0].page_content              # served, attributed context payload
docs[0].metadata["read_id"]       # -> cache.report_refusal()/report_success()
docs[0].metadata["sources"]       # artifact ids behind the read (provenance)
docs[0].metadata["cache_hit"]     # True == served with zero LLM spend
```

`include_evidence=True` additionally returns the retained raw evidence chunks as
separate `Document`s.

### 3. `CoalentVectorStoreRetriever` — your index as Coalent's substrate

Used internally by the factory; also available directly:

```python
from langchain_coalent import CoalentVectorStoreRetriever
retriever = CoalentVectorStoreRetriever(my_vectorstore, k=6)   # a Coalent Retriever
```

Document → Chunk mapping: `page_content` → `Chunk.text`; the artifact id (what
`cache.source_changed(...)` keys on) resolves as
`metadata["artifact_id"]` → `metadata["source"]` → `Document.id` → `metadata["id"]`
→ deterministic `chunk:<sha1(text)[:12]>` fallback. Give your documents a `source`
so invalidation has a stable identity. `metadata["version"]` → `Chunk.version`;
other metadata is not carried (Coalent's `Chunk` has no metadata dict).

### 4. The refusal loop (LangGraph-shaped)

When your answerer refuses over a served payload, that refusal is *evidence* —
hand the `read_id` back and Coalent serves the verbatim source excerpts the
extraction missed, then confirms the recovery as a durable alternate key:

```
retrieve ──> synthesize ──(refused?)──> report_refusal ──> re-synthesize ──> report_success
    ^                └─(answered)──> done                        └─(still refused)──> done
```

`examples/refusal_loop.py` is the runnable, fully offline demonstration. It is a
plain conditional loop implementing the identical LangGraph pattern (nodes +
one conditional edge) — `langgraph` is deliberately **not** a dependency of this
package; the example's docstring shows the 1:1 `StateGraph` mapping.

## Freshness — the reason this exists

```python
# Your ingestion pipeline noticed a document changed:
cache.source_changed("policy.md", text=new_text)   # surgical, provenance-keyed
# The very next retrieval that touches it rebuilds; untouched knowledge stays warm.
```

## Compatibility

Tested against `langchain-core` 1.x; written against the stable core contracts
(`BaseRetriever._get_relevant_documents`, `VectorStore.similarity_search`,
`Embeddings.embed_query/embed_documents`, `BaseChatModel.invoke`) which are
unchanged from 0.3, so `langchain-core>=0.3` is supported. Message text is read
from `message.content` directly (never `.text`), which is safe on both the 0.3
method form and the 1.x property form.
