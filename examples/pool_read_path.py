"""The v0.6 pool read path, end to end — pool serving, attribution headers, and the
refusal loop (``report_refusal`` / ``report_success``). Runs with **no API key**.

Run from the project root:

    python examples/pool_read_path.py

Shows:
  1. ``read_path="pool"``   — every read serves the budget-packed global fresh-claim pool
  2. ``pool_header``        — the measured golden path: a 3-line ``[title | source | date]``
                              callable wired to YOUR corpus metadata (0.73 vs 0.68 strict
                              accuracy for the bare default on a 605-question news benchmark)
  3. ``residual_spans``     — build-time capture of fact-bearing sentences the extractor missed
  4. the refusal loop       — your answerer refuses -> ``report_refusal(read_id)`` returns an
                              attributed retry payload -> the retry succeeds ->
                              ``report_success(read_id)`` confirms a durable query key

The synthesizer here is a deterministic stand-in that (deliberately) drops money sentences,
simulating real extraction loss; the embedder is a local bag-of-words stand-in so nothing
needs a network. In production use ``LLMSynthesizer(OpenAIProvider())`` (extractive by
default) and ``OpenAIEmbedder()`` — the pool path requires a semantic embedder and will
refuse to construct under the default lexical fallback.
"""
from __future__ import annotations

from coalent import (
    Chunk,
    FunctionEmbedder,
    HashingEmbedder,
    InMemoryRetriever,
    SemanticCache,
    Synthesis,
)

# --- Your corpus metadata, keyed by artifact id. The pool payload can only attribute
#     what it is given: wiring title/source/date here is the measured golden path. ---
DOC_META = {
    "news:acme": {"title": "Acme Robotics Series B", "source": "TechDaily", "date": "2026-05-14"},
    "news:orbit": {"title": "Orbit Analytics 4.2 release", "source": "CloudWire", "date": "2026-06-02"},
}


def pool_header(unit) -> str:
    """The 3-line golden-path header: ``[title | source | date]`` per source group."""
    aid = unit.evidence[0].artifact_id if unit.evidence else ""
    meta = DOC_META.get(aid)
    # NOTE: once a callable is set the built-in default does NOT take over on a miss —
    # always return your own fallback line instead of an empty string.
    if meta is None:
        return f"[source: {aid or unit.id}]"
    return f"[{meta['title']} | {meta['source']} | {meta['date']}]"


# --- A deterministic "extractive" synthesizer stand-in: one claim per sentence, but it
#     DROPS any sentence containing a dollar amount — simulating the extraction loss that
#     residual-span capture exists to catch. Swap for LLMSynthesizer in production. ---
class LossyExtractiveSynthesizer:
    def synthesize(self, query: str, chunks: list[Chunk]) -> Synthesis:
        claims: list[str] = []
        for chunk in chunks:
            for sentence in chunk.text.split(". "):
                sentence = sentence.strip().rstrip(".")
                if not sentence or "$" in sentence:   # the deliberate extraction loss
                    continue
                claims.append(sentence + ".")
        return Synthesis(
            understanding={"summary": "", "claims": claims, "entities": [], "facts": {}},
            used=list(range(len(chunks))),
        )


def main() -> None:
    retriever = InMemoryRetriever()
    retriever.add(
        "news:acme",
        "Acme Robotics closed a Series B round on May 14, raising $48 million led by "
        "Northbridge Capital. Acme Robotics builds warehouse automation robots. "
        "The company plans to double headcount by December.",
    )
    retriever.add(
        "news:orbit",
        "Orbit Analytics shipped version 4.2 of its dashboard product on June 2. "
        "The release adds real-time alerting and a new query planner.",
    )

    events: list[str] = []
    cache = SemanticCache(
        retriever,
        LossyExtractiveSynthesizer(),
        # Demo-only local embedder: bag-of-words via FunctionEmbedder. The pool path
        # refuses the raw HashingEmbedder default because claim cosine collapses to
        # keyword overlap — use OpenAIEmbedder() or a real local model in production.
        embedder=FunctionEmbedder(HashingEmbedder(dim=512).embed),
        read_path="pool",            # v0.6: serve the global fresh-claim pool
        pool_header=pool_header,     # the measured attribution golden path
        residual_spans=True,         # capture what the extractor missed, serve on refusal
        query_keys=True,             # confirmed rescues earn durable alternate keys
        # span_margin raised for the demo ONLY so the captured span is NOT served
        # proactively (with the default 0.0 it would be, and the refusal loop below would
        # never be needed) — leave it at the default in production.
        span_margin=0.75,
        on_event=lambda event: events.append(event["event"]),
    )

    # 1) A read builds understanding lazily, then serves the packed, attributed pool.
    r1 = cache.get("what does Acme Robotics build?")
    print("1) pool payload (note the [title | source | date] headers):\n")
    print(r1.context["pool"])

    # 2) Ask about the fact the extractor dropped. The payload cannot contain it.
    question = "how much did Acme Robotics raise in its Series B?"
    r2 = cache.get(question)
    payload = r2.context["pool"]
    answer = answerer(question, payload)
    print(f"\n2) first pass -> {'answered' if answer else 'REFUSED (fact not in payload)'}")

    # 3) The refusal loop: a refusal is evidence. Hand back the read_id; the cache
    #    re-scores the captured residual spans and returns an attributed retry payload.
    if answer is None:
        retry = cache.report_refusal(r2.read_id)
        if retry is not None:
            print(f"\n3) retry payload from report_refusal:\n{retry}")
            answer = answerer(question, payload + "\n" + retry)
            if answer is not None:
                # Confirm ONLY a retry that actually succeeded: this makes the rescue's
                # provisional query key durable and marks the unit for append-only repair.
                cache.report_success(r2.read_id)
    print(f"\n4) final answer: {answer}")

    print(f"\nevents: {sorted(set(events))}")


def answerer(question: str, payload: str) -> str | None:
    """A stand-in for YOUR answer model: it can only answer from the payload, and it
    refuses when the needed fact is absent. Replace with your LLM + refusal detector."""
    for line in payload.splitlines():
        if "$" in line:
            return next(word for word in line.split() if word.startswith("$")) + " (from: " + line.strip() + ")"
    return None


if __name__ == "__main__":
    main()
