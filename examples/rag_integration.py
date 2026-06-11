"""How to retrofit Coalent into an existing RAG app.

Five adoption steps:
  1. Wrap your vector DB as a Retriever (you keep doing the retrieval).
  2. Inject your LLM via a Synthesizer (StubSynthesizer here — no key needed).
  3. Build the cache.
  4. Swap your retrieve-and-stuff step for ``cache.get(...)``.
  5. Wire invalidation from your webhooks / write path.

Run: python examples/rag_integration.py
"""
from __future__ import annotations

import re

from coalent import Chunk, SemanticCache, StubSynthesizer


# --- Step 1: wrap YOUR vector DB. This stand-in does keyword-ranked top-k, but the
#     contract is all that matters: retrieve() runs your search and returns Chunks
#     carrying provenance (artifact_id + version). ``Retriever`` is a structural
#     protocol — no base class to inherit. ---
class MiniVectorRetriever:
    def __init__(self) -> None:
        self._docs: list[tuple[str, str, str]] = []

    def add(self, artifact_id: str, text: str, *, version: str = "v1") -> None:
        self._docs.append((artifact_id, text, version))

    def retrieve(self, query: str, *, namespace: str | None = None, k: int = 4) -> list[Chunk]:
        terms = set(re.findall(r"\w+", query.lower()))
        scored: list[tuple[int, str, str, str]] = []
        for artifact_id, text, version in self._docs:
            score = len(terms & set(re.findall(r"\w+", text.lower())))
            if score:
                scored.append((score, artifact_id, text, version))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [Chunk(a, t, version=v) for _, a, t, v in scored[:k]]


def main() -> None:
    # Step 1: your retriever.
    retriever = MiniVectorRetriever()
    retriever.add("confluence:hr", "Leave policy: 21 days of annual leave per year.")
    retriever.add("confluence:remote", "Remote work is allowed up to three days per week.")

    # Step 2 + 3: inject your LLM via a Synthesizer (Stub here) and build the cache.
    cache = SemanticCache(retriever, StubSynthesizer())

    # Step 4: replace "retrieve top-k -> stuff prompt" with one call.
    ctx = cache.get("what is our leave policy?")
    print("cold :", ctx.cache_hit, "|", ctx.context["understanding"].get("summary"))

    # asked differently, same meaning -> a warm hit (semantic reuse, no re-synthesis)
    ctx = cache.get("tell me about the leave policy")
    print("warm :", ctx.cache_hit)

    # Step 5: wire invalidation from your webhook / write path.
    cache.source_changed("confluence:hr", text="Leave policy: now 25 days of annual leave per year.")
    ctx = cache.get("what is our leave policy now?")
    print("fresh:", ctx.cache_hit, "(re-materialized after the change)")


if __name__ == "__main__":
    main()
