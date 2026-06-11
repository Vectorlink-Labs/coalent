"""Runnable end-to-end demo of the cognitive cache.

Run from the project root:

    python examples/quickstart.py

Shows: cold miss -> warm hit -> a source change -> surgical invalidation ->
fresh re-materialization. Uses ``StubSynthesizer``, so it runs with no API key.
"""
from __future__ import annotations

from coalent import InMemoryRetriever, SemanticCache, StubSynthesizer


def main() -> None:
    # 1. Any retriever — your vector DB / tool / API plugs in here.
    retriever = InMemoryRetriever()
    retriever.add("confluence:hr", "Leave policy: 21 days of annual leave per year.")

    # 2. Build the cache. Swap StubSynthesizer for LLMSynthesizer(OpenAIProvider()) in prod.
    cache = SemanticCache(retriever, StubSynthesizer())

    q = "what is our leave policy?"

    r1 = cache.get(q)
    print(f"1) cold miss     -> cache_hit={r1.cache_hit}  understanding={r1.context['understanding']}")

    r2 = cache.get(q)
    print(f"2) warm hit      -> cache_hit={r2.cache_hit}")

    # 3. A source changed -> only the units that used it go dirty.
    res = cache.source_changed("confluence:hr", text="Leave policy: now 25 days of annual leave.")
    print(f"3) source change -> dirtied={res.dirtied}")

    r3 = cache.get(q)
    print(f"4) post-change   -> cache_hit={r3.cache_hit} (re-materialized fresh)")

    print(f"\nstats: {cache.stats()}")


if __name__ == "__main__":
    main()
