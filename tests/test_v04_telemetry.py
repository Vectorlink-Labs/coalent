"""v0.4 — usage/cost telemetry: token accounting flows provider -> synthesizer -> Result/stats.

Closes the real deployment gap where teams bypassed LLMSynthesizer (losing the cache knobs +
containment) just to read token usage. A provider may return a Generation(text, usage); a bare
str still works (usage=None), so custom providers never break.
"""
from __future__ import annotations

from coalent import Generation, SemanticCache, SQLiteCognitionStore, StubSynthesizer, Usage
from coalent.semantic import (
    Chunk,
    InMemoryRetriever,
    JSONPassthroughSynthesizer,
    LLMSynthesizer,
    Synthesis,
)


class _UsageProvider:
    """Returns Generation with usage — like the shipped OpenAI/Anthropic adapters."""

    def generate(self, *, model, system, user, max_tokens, temperature):  # type: ignore[no-untyped-def]
        return Generation(
            text='{"summary": "ok", "claims": ["c"], "used": [0]}',
            usage=Usage(prompt_tokens=100, completion_tokens=20, model=model),
        )


class _BareStrProvider:
    """Legacy/custom provider returning a plain str — must keep working (usage=None)."""

    def generate(self, *, model, system, user, max_tokens, temperature):  # type: ignore[no-untyped-def]
        return '{"summary": "ok", "claims": ["c"], "used": [0]}'


def _retriever() -> InMemoryRetriever:
    r = InMemoryRetriever(top_k=1)
    r.add("doc:1", "leave policy grants twenty days")
    return r


def test_usage_flows_to_result_and_stats() -> None:
    # hit_threshold low so the identical second query re-hits the unit (the stub understanding
    # is query-independent, so the match leans on the seed-query term — see _match_score).
    cache = SemanticCache(_retriever(), LLMSynthesizer(_UsageProvider()), hit_threshold=0.2)

    first = cache.get("leave policy")          # MISS -> synthesizes -> usage present
    assert first.cache_hit is False
    assert first.usage is not None
    assert first.usage.prompt_tokens == 100
    assert first.usage.completion_tokens == 20
    assert first.usage.total_tokens == 120

    second = cache.get("leave policy")         # HIT -> no LLM call -> zero/None usage (the saving)
    assert second.cache_hit is True
    assert second.usage is None

    s = cache.stats()
    assert s["synth_calls"] == 1
    assert s["synth_tokens"] == 120
    assert s["avg_synth_tokens"] == 120.0
    assert s["tokens_saved"] == 120            # the one HIT credited its unit's build cost


def test_bare_str_provider_still_works() -> None:
    """Backward-compat: a provider returning a plain str must not break (usage just None)."""
    cache = SemanticCache(_retriever(), LLMSynthesizer(_BareStrProvider()))
    result = cache.get("leave policy")
    assert result.usage is None
    assert cache.stats()["synth_calls"] == 0   # no usage reported -> not counted


def test_stub_synthesizer_has_no_usage() -> None:
    cache = SemanticCache(_retriever(), StubSynthesizer())
    result = cache.get("leave policy")
    assert result.usage is None
    assert cache.stats()["synth_tokens"] == 0


def test_usage_accumulates_across_retries() -> None:
    """A malformed first response forces a retry; usage must sum both calls."""
    class _FlakyProvider:
        def __init__(self) -> None:
            self.n = 0

        def generate(self, *, model, system, user, max_tokens, temperature):  # type: ignore[no-untyped-def]
            self.n += 1
            text = "not json" if self.n == 1 else '{"summary": "ok", "used": [0]}'
            return Generation(text=text, usage=Usage(prompt_tokens=50, completion_tokens=10))

    synth = LLMSynthesizer(_FlakyProvider(), retries=1)
    out: Synthesis = synth.synthesize("q", [Chunk(artifact_id="a", text="t")])
    assert out.ok is True
    assert out.usage is not None
    assert out.usage.prompt_tokens == 100      # 50 + 50 across the two calls
    assert out.usage.completion_tokens == 20


class _AlwaysBadJson:
    """Always returns unparseable text WITH usage — failed synthesis still costs real tokens."""

    def generate(self, **kw):  # type: ignore[no-untyped-def]
        return Generation(text="not json", usage=Usage(prompt_tokens=50, completion_tokens=10))


def test_failed_synthesis_still_bills_usage() -> None:
    out = LLMSynthesizer(_AlwaysBadJson(), retries=1).synthesize("q", [Chunk(artifact_id="a", text="t")])
    assert out.ok is False
    assert out.understanding.get("_synthesis_failed") is True
    assert out.usage is not None and out.usage.total_tokens == 120   # both billed attempts

    cache = SemanticCache(_retriever(), LLMSynthesizer(_AlwaysBadJson(), retries=1))
    res = cache.get("leave policy")
    assert res.usage is not None and res.usage.total_tokens == 120   # accounted despite failure
    assert cache.stats()["synth_calls"] == 1
    assert cache.stats()["synth_tokens"] == 120


def test_stale_rematerialize_bills_usage() -> None:
    cache = SemanticCache(_retriever(), LLMSynthesizer(_UsageProvider()), hit_threshold=0.2)
    cache.get("leave policy")                                         # MISS -> synth_calls 1
    cache.source_changed("doc:1", text="leave policy now grants thirty days")  # dirty the unit
    result = cache.get("leave policy")                               # STALE -> re-materialize
    assert result.cache_hit is False
    assert result.usage is not None and result.usage.total_tokens == 120
    assert cache.stats()["synth_calls"] == 2
    assert cache.stats()["synth_tokens"] == 240


def test_passthrough_synthesizer_reports_no_usage() -> None:
    retriever = InMemoryRetriever(top_k=1)
    retriever.add("api:1", '{"status": "ok", "limit": 100}')
    cache = SemanticCache(retriever, JSONPassthroughSynthesizer())
    result = cache.get("status")
    assert result.usage is None
    assert cache.stats()["synth_calls"] == 0
    assert cache.stats()["synth_tokens"] == 0


def test_tokens_saved_survives_store_reload(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The headline savings number must NOT zero out after a restart: a HIT credits the reloaded
    unit's PERSISTED build cost, not an in-process running average."""
    db = str(tmp_path / "tele.db")
    store1 = SQLiteCognitionStore(db)
    first = SemanticCache(_retriever(), LLMSynthesizer(_UsageProvider()), hit_threshold=0.2, store=store1)
    first.get("leave policy")                                        # MISS -> unit.synth_tokens=120, persisted
    store1.close()

    store2 = SQLiteCognitionStore(db)                               # "restart": counters reset, units reload
    second = SemanticCache(_retriever(), LLMSynthesizer(_UsageProvider()), hit_threshold=0.2, store=store2)
    result = second.get("leave policy")                            # HIT on the reloaded unit
    assert result.cache_hit is True
    assert second.stats()["tokens_saved"] == 120                   # credited from the reloaded unit's cost
    store2.close()


def test_exports_available_from_both_surfaces() -> None:
    import coalent
    from coalent import Generation as G1
    from coalent import Usage as U1
    from coalent.semantic import Generation as G2
    from coalent.semantic import Usage as U2

    assert "Usage" in coalent.__all__ and "Generation" in coalent.__all__
    assert (G1, U1) == (G2, U2)
