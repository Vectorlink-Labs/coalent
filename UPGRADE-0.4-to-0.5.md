# Upgrading from 0.4 to 0.5

**Nothing breaks.** Every v0.5 feature is additive and default-OFF (or ``fast="auto"``, which
is equivalence-pinned to the pure path). v0.4 code runs identically on v0.5.

## Recommended adoption path

```python
cache = SemanticCache(
    retriever, synthesizer, embedder=OpenAIEmbedder(),
    preset="multi_hop",           # arms cross-unit recall + the hop-2 bridge
    widen_chunks=24,              # build from the source, not the retrieval keyhole
    provenance_admission=True,    # never build duplicate understanding
    adaptive_hit=True,            # hit gate self-calibrates as the cache grows
)
```

Then, optionally, the experimental v0.6 read-path preview (measured: ties naive dense RAG's
best arm at 0.79x its tokens, held-out n=605):

```python
cache = SemanticCache(..., serve="pool", serve_budget=600,
                      pool_header=lambda u: f"[{u.id}]")
# renderers read result.context["pool"]
```

## Deprecations
- ``select_floor`` is deprecated (superseded by pool serving); it still works in 0.5.

## Notes
- ``fast="auto"``: with numpy installed (``pip install "coalent[fast]"``) the read path runs
  vectorized twins of the matching/recall/bridge scans — identical results, pinned by tests.
- Widening needs a source: either your retriever grows a duck-typed
  ``widen(artifact_id, *, limit=None)`` method, or pass ``source_fetcher=...``. Without
  either, widening warns once and builds from the retrieved chunks (v0.4 behavior).
