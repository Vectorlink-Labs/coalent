"""v0.4 additive mechanisms — logic tier (hermetic, deterministic).

Pins the four packaging changes that turn the proven extractive approach into first-class
library knobs, each with a controllable synonym-axis embedder so the cosines are exact and
hand-checkable (embedding *quality* is validated separately on real OpenAI at the gate):

  1. ``extract=True``            -> the QUERY-INDEPENDENT extractive instruction (synth level)
  2. ``select_floor``           -> serve atoms by MEANING (per-claim cosine), not keyword trim
  3. ``residual_floor``         -> retain number-bearing cited spans the extractor dropped
  4. ``hit_margin``             -> don't commit to a unit that only ties a topical neighbour

Default posture (v0.4): ``extract`` is ON by default (strictly better — keeps every number);
``select_floor`` / ``residual_floor`` / ``hit_margin`` stay OFF (situational, can cost) and each
has a default-OFF invariant proven below — unset, that knob's read is v0.3. ``extract=False``
restores the v0.3 prose path.
"""
from __future__ import annotations

from coalent import EXTRACTIVE_INSTRUCTION, FunctionEmbedder, StubProvider
from coalent.semantic import (
    Chunk,
    InMemoryRetriever,
    LLMSynthesizer,
    SemanticCache,
    SQLiteCognitionStore,
    Synthesis,
)
from coalent.semantic.synthesizer import _DEFAULT_INSTRUCTION

# One meaning axis per content word -> exact cosines. Stopwords ("the", "is", "in") are NOT
# axes, so they don't perturb similarity (matches how a real embedder ignores filler).
_AXES = (
    "alice", "bob", "france", "germany", "capital", "paris", "berlin",
    "leads", "team", "limit", "days", "21", "sick", "leave",
)


def _embed(text: str) -> list[float]:
    words = set(text.lower().replace("?", " ").replace(";", " ").replace(".", " ").split())
    v = [1.0 if a in words else 0.0 for a in _AXES]
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v] if norm else v


def _fe() -> FunctionEmbedder:
    return FunctionEmbedder(_embed)


class _AtomSynth:
    """LLM stand-in: '; '-separated atomic claims, summary = the full source text."""

    def synthesize(self, query: str, chunks: list[Chunk]) -> Synthesis:
        text = " ".join(c.text for c in chunks)
        claims = [s.strip() for s in text.split(";") if s.strip()]
        return Synthesis(
            understanding={"summary": text, "claims": claims},
            used=list(range(len(chunks))),
        )


# ============================================================ 1. extract=True (synth level)
def test_extract_true_selects_the_extractive_instruction() -> None:
    synth = LLMSynthesizer(StubProvider(canned="{}"), extract=True)
    assert synth._instruction is EXTRACTIVE_INSTRUCTION


def test_extract_default_is_the_extractive_instruction() -> None:
    """v0.4: extract defaults ON — the query-independent extractive instruction is now the default."""
    synth = LLMSynthesizer(StubProvider(canned="{}"))
    assert synth._instruction is EXTRACTIVE_INSTRUCTION


def test_extract_false_restores_the_v03_prose_instruction() -> None:
    """The escape hatch: extract=False gives back the exact v0.3 query-grounded prose summary."""
    synth = LLMSynthesizer(StubProvider(canned="{}"), extract=False)
    assert synth._instruction is _DEFAULT_INSTRUCTION


def test_explicit_instruction_wins_over_extract() -> None:
    """A caller who passes their own instruction is never silently overridden by extract=True."""
    synth = LLMSynthesizer(StubProvider(canned="{}"), extract=True, instruction="my own rules")
    assert synth._instruction == "my own rules"


# ============================================================ 2. select_floor (serve by meaning)
def _select_cache(**kw: object) -> SemanticCache:
    retriever = InMemoryRetriever(top_k=1)
    # Two claims that BOTH contain 'alice' -> lexical trim can't tell them apart; only a graded
    # cosine can. 'alice france' is tight (cos 0.71); the diluted claim is loose (cos 0.45).
    retriever.add("d", "alice france; alice bob capital france germany")
    cache = SemanticCache(
        retriever, _AtomSynth(), embedder=_fe(),
        hit_threshold=0.2, enable_coverage_escalation=False, read_path="unit",
        **kw,  # type: ignore[arg-type]
    )
    cache.get("alice france alice bob capital france germany")  # build the unit
    return cache


def test_select_floor_serves_only_atoms_above_the_cosine_floor() -> None:
    served = _select_cache(select_floor=0.5).get("alice")
    assert served.cache_hit is True
    # The diluted 'alice ...' claim (cos 0.45 < 0.5) is dropped even though it CONTAINS 'alice'.
    assert served.context["understanding"]["claims"] == ["alice france"]


def test_no_select_floor_is_the_v03_lexical_projection() -> None:
    """Default (select_floor=None): the lexical trim keeps every claim that CONTAINS the term,
    including the diluted one a cosine floor would drop -> proves the knob changed the behaviour."""
    served = _select_cache().get("alice")
    assert served.cache_hit is True
    assert set(served.context["understanding"]["claims"]) == {
        "alice france", "alice bob capital france germany",
    }


# ============================================================ 3. residual_floor (safety net)
class _LossySynth:
    """An extractor that DROPS a fact: it keeps only the first sentence and omits the rest —
    so a number that lives in a later sentence is missing from the claims (the failure mode)."""

    def synthesize(self, query: str, chunks: list[Chunk]) -> Synthesis:
        text = " ".join(c.text for c in chunks)
        first = text.split(".")[0].strip()
        return Synthesis(understanding={"summary": first, "claims": [first]},
                         used=list(range(len(chunks))))


def _residual_cache(**kw: object) -> tuple[SemanticCache, str]:
    retriever = InMemoryRetriever(top_k=1)
    retriever.add("policy", "alice leads team. limit 21 days sick leave.")  # number in sentence 2
    cache = SemanticCache(
        retriever, _LossySynth(), embedder=_fe(),
        hit_threshold=0.2, enable_coverage_escalation=False, read_path="unit",
        **kw,  # type: ignore[arg-type]
    )
    result = cache.get("alice leads team")
    return cache, result.unit_id


def test_residual_floor_retains_the_dropped_number_span() -> None:
    cache, uid = _residual_cache(residual_floor=0.3)
    claims = [str(c) for c in cache._units[uid].understanding["claims"]]
    # The extractor dropped "limit 21 days sick leave"; the residual net puts it back as an atom.
    assert any("21 days" in c for c in claims)
    # ...and it was embedded (parallel to claim_texts) so it participates in coverage/recall.
    assert len(cache._units[uid].claim_embeddings) == len(cache._units[uid].understanding["claims"]) + 1


def test_no_residual_floor_leaves_the_gap() -> None:
    cache, uid = _residual_cache()  # residual_floor=None
    claims = [str(c) for c in cache._units[uid].understanding["claims"]]
    assert not any("21 days" in c for c in claims)  # the dropped number stays dropped


def test_residual_atoms_survive_a_store_reload_parallel_to_embeddings(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The invariant most likely to silently rot: after persist->reload, the residual atoms AND
    their embeddings must stay exactly parallel (claim_embeddings == claims + summary), so serve /
    coverage / recall keep zipping the right text to the right vector."""
    db = str(tmp_path / "cog.db")
    retriever = InMemoryRetriever(top_k=1)
    retriever.add("policy", "alice leads team. limit 21 days sick leave.")

    store1 = SQLiteCognitionStore(db)
    cache1 = SemanticCache(
        retriever, _LossySynth(), embedder=_fe(), hit_threshold=0.2,
        enable_coverage_escalation=False, residual_floor=0.3, store=store1, read_path="unit",
    )
    uid = cache1.get("alice leads team").unit_id
    store1.close()

    # Fresh cache over the same DB — units are reloaded from serde, not rebuilt.
    cache2 = SemanticCache(
        retriever, _LossySynth(), embedder=_fe(), hit_threshold=0.2,
        store=SQLiteCognitionStore(db), read_path="unit",
    )
    unit = cache2._units[uid]
    claim_texts = cache2._claim_texts(unit.understanding)          # [claims..., summary]
    assert any("21 days" in t for t in claim_texts)                # residual atom persisted
    assert len(unit.claim_embeddings) == len(claim_texts)          # STILL parallel after reload
    # and the semantic-select path (which zips the two) works on the reloaded unit
    cache2._select_floor = 0.0
    qe = tuple(_fe().embed("21 days sick"))
    assert any("21 days" in t for t in cache2._select_claim_texts(qe, unit))


# ============================================================ 4. hit_margin (precision guard)
def _two_unit_cache(**kw: object) -> SemanticCache:
    """Two units with ORTHOGONAL seeds (so neither build-query hits the other), each sharing
    exactly one token with the probe 'paris berlin' -> the probe ties them 0.5 vs 0.5, which is
    exactly the ambiguous routing case hit_margin should catch."""
    retriever = InMemoryRetriever(top_k=1)
    cache = SemanticCache(
        retriever, _AtomSynth(), embedder=_fe(), hit_threshold=0.35, read_path="unit",
        **kw,  # type: ignore[arg-type]
    )
    retriever.add("d1", "france paris")
    cache.get("france paris")
    retriever.add("d2", "germany berlin")
    cache.get("germany berlin")           # orthogonal to d1 -> births its own unit
    return cache


def test_hit_margin_refuses_to_commit_to_an_ambiguous_tie() -> None:
    cache = _two_unit_cache(hit_margin=0.1)
    before = len(cache._units)
    result = cache.get("paris berlin")    # ties d1 and d2 (best - second ~ 0 < 0.1)
    assert result.cache_hit is False      # ambiguous -> do not commit to a neighbour
    assert len(cache._units) == before + 1  # materialised the query's own unit instead


def test_no_hit_margin_commits_to_the_best_match() -> None:
    cache = _two_unit_cache()             # hit_margin defaults to 0.0 (v0.3 behaviour)
    before = len(cache._units)
    result = cache.get("paris berlin")
    assert result.cache_hit is True       # commits to the top unit
    assert len(cache._units) == before    # no new unit born


def test_hit_margin_never_blocks_a_lone_unit() -> None:
    """With one unit there is no runner-up, so the margin must not suppress a genuine hit."""
    retriever = InMemoryRetriever(top_k=1)
    cache = SemanticCache(
        retriever, _AtomSynth(), embedder=_fe(), hit_threshold=0.35, hit_margin=0.9,
        read_path="unit",
    )
    retriever.add("d", "france paris")
    cache.get("france paris")
    assert cache.get("paris").cache_hit is True
