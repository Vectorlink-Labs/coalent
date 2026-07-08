"""SemanticCache — the embedding-keyed read path.

One method: ``get(query)``. It embeds the query, finds an existing fresh unit by
cosine similarity (a semantic cache hit), and otherwise retrieves + synthesizes a
new one — always retaining the raw evidence so it can never return less than plain
retrieval. Source changes mark units dirty via provenance; they re-materialize
lazily on the next matching read.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..domain.models import ChangeEvent, ProvenanceManifest, SourceSpan
from .embedding import (
    Embedder,
    HashingEmbedder,
    cosine,
    default_embedder,
    default_thresholds_for,
    embed_texts,
    tokenize,
)
from .ports import Chunk, Retriever, Synthesizer, Usage
from .store import CognitionStore
from .unit import Cognition

logger = logging.getLogger(__name__)

_NUM = re.compile(r"\d")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _number_sentences(chunks: list[Chunk]) -> list[str]:
    """Number-bearing sentences from ``chunks`` (the dropped-fact failure is numeric),
    de-duplicated, order preserved — the residual safety net's candidate spans."""
    out: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        for raw in _SENT_SPLIT.split(chunk.text):
            s = raw.strip()
            if s and _NUM.search(s) and s not in seen:
                seen.add(s)
                out.append(s)
    return out


@dataclass(slots=True)
class Related:
    """A cross-unit relationship surfaced on a read (shared entity or source)."""

    unit_id: str
    understanding: dict[str, Any]
    evidence: list[Chunk]
    relation: str   # "shared_entity" | "shared_source"
    score: float    # relevance of this related unit to the current query


@dataclass(slots=True)
class RecalledClaim:
    """An atomic claim surfaced by cross-unit recall (v0.4): a fact pooled from ANY
    cached unit by claim-level meaning — not just the single best-match unit. The
    substrate that recovers an answer-claim whose unit ranks below top-k."""

    claim: str
    score: float          # cosine(query, this claim's embedding)
    unit_id: str          # the cached unit this claim came from
    source: str = ""      # only set when recall_raw is opted in (default OFF — keeps tokens compact)


@dataclass(slots=True)
class Result:
    """What a read returns: understanding + retained raw evidence + related units."""

    understanding: dict[str, Any]
    evidence: list[Chunk]
    cache_hit: bool
    unit_id: str
    confidence: float
    namespace: str
    related: list[Related] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)  # minimum decision-relevant payload
    coverage: float = 1.0                                   # how well the unit covers the query
    escalated: bool = False                                 # had to pull fresh raw for this query
    recalled: list[RecalledClaim] = field(default_factory=list)  # v0.4 cross-unit atomic claims
    usage: Usage | None = None  # synth tokens for THIS read; None when no LLM ran (a hit, OR a
    #                             usage-less provider) — use `cache_hit` as the authoritative hit signal
    needs_retrieval: bool = False  # S3 hint: cache under-covered even after recall — you MAY retry/widen

    @property
    def raw_text(self) -> str:
        """The retained raw evidence as text — the detail the LLM may need."""
        return "\n\n".join(chunk.text for chunk in self.evidence)


@dataclass(slots=True)
class InvalidationResult:
    """Outcome of applying one change event."""

    dirtied: list[str] = field(default_factory=list)
    skipped_unchanged: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    matched_units: int = 0


class ContextStrategy:
    """How much to place in the returned context payload (raw stays reachable)."""

    CONTEXT_FIRST = "context_first"  # understanding; raw only when escalated (default)
    CONTEXT_RAW = "context_raw"      # understanding + raw, always
    CONTEXT_ONLY = "context_only"    # understanding only


@dataclass(slots=True)
class FreshnessPolicy:
    """Time-based freshness for feed-less sources (APIs / tools).

    On expiry (``max_age`` seconds since last fresh), the cache revalidates by
    re-fetching + hashing via ``revalidate(artifact_id) -> (text, version) | None``:
    unchanged content stays fresh (no rebuild — content_hash earns its keep),
    changed content re-materializes. With no ``revalidate``, expiry conservatively
    rebuilds on the next read.
    """

    max_age: float | None = None
    revalidate: Callable[[str], "tuple[str, str] | None"] | None = None


class SemanticCache:
    """Embedding-keyed cognitive cache over understanding + raw evidence.

    **The read flow (every knob is additive; the defaults reproduce v0.3):**

    1. *Match* a unit by meaning (``0.7·cos(query, understanding) + 0.3·cos(query, seed)``).
    2. *Cover* — the query's max per-claim cosine to the unit's atoms.
    3. *Recall* (``cross_unit_recall``) — if under-covered, MaxSim atoms from OTHER units.
    4. *Escalate* (``enable_coverage_escalation``) — if the cache STILL under-covers, retrieve
       fresh raw: the **RAG floor**. No LLM call; identical to plain RAG on that query.

    **Getting the most from the cache (structured / reuse-heavy docs — policies, specs, FAQs):**

    - ``extract=True`` on :class:`LLMSynthesizer` — build QUERY-INDEPENDENT atoms so one cached
      unit answers many later questions (the reuse win). See ``EXTRACTIVE_INSTRUCTION``.
    - ``select_floor`` — serve the matched unit's atoms by *meaning* (per-claim cosine ≥ floor)
      instead of the lexical keyword trim: the query-relevant facts, fewer tokens.
    - ``residual_floor`` — at build, retain number-bearing cited spans the extractor missed as
      extra atoms, closing the extractor-recall gap. Embedding-only, no extra LLM call.

    **The reliability gate (``coverage_scorer``, "S2").** OPTIONAL and OFF by default — cosine
    coverage alone is a heuristic that can rate an *adjacent* unit as covering. Supply a
    cross-encoder / NLI / LLM-entailment scorer to get a **hard containment guarantee**: it is
    consulted only in the ambiguous band ``[coverage_floor, coverage_ceiling)`` and decides whether
    the cache truly answers or must fall to the RAG floor. It costs one judge call per ambiguous
    read, so it is opt-in — but if you need "never silently serves less than RAG" as a guarantee
    rather than a tendency, S2 is the mechanism that provides it.
    """

    def __init__(
        self,
        retriever: Retriever,
        synthesizer: Synthesizer,
        *,
        embedder: Embedder | None = None,
        hit_threshold: float | None = None,
        hit_margin: float = 0.0,
        coverage_floor: float | None = None,
        understanding_weight: float = 0.7,
        route_by_claim: bool = False,
        learn_behavior: bool = True,
        max_hit_queries: int = 16,
        enable_coverage_escalation: bool = True,
        learn_on_escalation: bool = False,
        coverage_scorer: Callable[[str, dict[str, Any]], float] | None = None,
        coverage_ceiling: float = 1.0,
        relevance_gate: Callable[[str, list[Chunk]], list[Chunk]] | None = None,
        cross_unit_recall: bool = True,   # DEFAULT since v0.4 (multi-hop; free on single-hop)
        recall_threshold: float | None = None,
        recall_limit: int = 6,
        recall_raw: bool = False,
        select_floor: float | None = None,
        residual_floor: float | None = None,
        residual_limit: int = 24,
        strategy: str = ContextStrategy.CONTEXT_FIRST,
        store: CognitionStore | None = None,
        freshness: FreshnessPolicy | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._retriever = retriever
        self._synth = synthesizer
        self._embedder: Embedder = embedder if embedder is not None else default_embedder()
        # Thresholds derive from the embedder when unset: OpenAI cosines are compressed
        # (~0.33), the lexical HashingEmbedder scores higher (~0.6) — one fixed default
        # can't fit both. Override explicitly, or tune via coalent.calibrate_thresholds.
        hit_default, cov_default = default_thresholds_for(self._embedder)
        self._threshold = hit_default if hit_threshold is None else hit_threshold
        # Precision guard (default 0.0 = off, v0.3 behavior): require the top unit to beat the
        # runner-up by this cosine margin before committing to it as a hit — disambiguates a query
        # sitting between topically-adjacent units (the multi-doc over-merge at compressed cosines).
        self._hit_margin = hit_margin
        self._coverage_floor = cov_default if coverage_floor is None else coverage_floor
        self._understanding_weight = understanding_weight
        self._route_by_claim = route_by_claim
        self._learn_behavior = learn_behavior
        self._max_hit_queries = max_hit_queries
        self._enable_coverage_escalation = enable_coverage_escalation
        # v0.4 (opt-in, OFF) — on escalation, also synthesize the fresh evidence into a NEW unit so
        # the cache LEARNS (compounding); costs one synthesis per escalation, hence default off.
        self._learn_on_escalation = learn_on_escalation
        self._coverage_scorer = coverage_scorer
        self._coverage_ceiling = coverage_ceiling
        self._relevance_gate = relevance_gate
        # v0.4 — cross-unit claim recall (opt-in; semantic-embedder only). The recall
        # TRIGGER is its own tunable threshold; None falls back to coverage_floor. It is
        # meant to be swept on the eval (M0) to the fewest-misses point — see get().
        self._cross_unit_recall = cross_unit_recall
        self._recall_threshold = recall_threshold
        self._recall_limit = recall_limit
        # DEFAULT OFF — protects the token win. Raw spans recover facts a lossy claim dropped but
        # ~double the tokens (which would defeat the cache's whole purpose), so the default path
        # serves only compact claims+summary; flip this on only to trade tokens for accuracy.
        self._recall_raw = recall_raw
        # v0.4 — semantic-select serve: when set, serve the matched unit's atoms whose per-claim
        # cosine to the query >= select_floor (the query-RELEVANT atoms by meaning) instead of the
        # lexical keyword trim. None = the v0.3 lexical projection (backward compatible).
        self._select_floor = select_floor
        # v0.4 — residual safety net: at build, retain number-bearing source sentences the extractor
        # did NOT capture (max per-claim cosine < residual_floor) as extra verbatim atoms bound to the
        # unit — closes the extractor-recall gap for dropped numbers. Embedding-only, no extra LLM
        # call. None = off (backward compatible). Lower floor = retain fewer (only clearly-missed).
        self._residual_floor = residual_floor
        # Bound the safety net so it can't defeat the token win on a big/numeric doc: keep only the
        # N LEAST-covered missed spans (ranked most-missed first). <= 0 = unlimited (escape hatch).
        self._residual_limit = residual_limit
        self._strategy = strategy
        self._store = store
        self._freshness = freshness
        self._clock = clock
        self._units: dict[str, Cognition] = {}
        self._artifact_index: dict[str, set[str]] = {}
        self._entity_index: dict[str, set[str]] = {}
        # Read-time observability (escalation rate is the "am I drifting to RAG?" signal).
        self._reads_total = 0
        self._reads_hit = 0
        self._reads_escalated = 0
        # Token/cost telemetry — so the saving (a HIT spends ZERO synthesis tokens) is measurable
        # end-to-end without bypassing this synthesizer. Only counts calls that carried usage.
        self._synth_calls = 0
        self._synth_prompt_tokens = 0
        self._synth_completion_tokens = 0
        self._synth_cost = 0.0
        # Reload-safe savings: each HIT credits the hit unit's persisted build cost (synth_tokens),
        # so the number is correct after a restart AND under mixed (usage-less) providers.
        self._tokens_saved = 0
        # Restart-safe: load persisted units, backfill pre-v0.3 ones (compute their
        # understanding/claim embeddings once), and rebuild the invalidation indexes.
        if store is not None:
            for unit in store.all():
                self._units[unit.id] = unit
                if unit.needs_backfill and unit.understanding:
                    self._backfill_cognition(unit)
                self._reindex(unit)

    def _persist(self, unit: Cognition) -> None:
        if self._store is not None:
            self._store.put(unit)

    def _mark_fresh(self, unit: Cognition) -> None:
        unit.mark_fresh()
        unit.freshness_epoch = self._clock()

    # ---------------------------------------------------------------- read
    def get(
        self,
        query: str,
        *,
        namespace: str | None = None,
        related: int = 3,
        strategy: str | None = None,
    ) -> Result:
        """Fetch fresh, decision-ready context for a query. The one read method.

        Returns the minimum decision-relevant ``context`` for this query (raw stays
        reachable via ``evidence`` / ``drill``). A cache hit that under-covers the
        query auto-escalates to fresh raw — no manual signal. ``related`` folds in
        up to N related units; ``strategy`` overrides the context payload policy.
        """
        strat = strategy or self._strategy
        ns = namespace or ""
        qe = tuple(self._embedder.embed(query))
        read_usage: Usage | None = None  # synth cost of THIS read; None on a hit OR a usage-less provider

        best_id, best_score, second_score = self._best_match(qe, ns)
        # hit_margin (default 0.0 = off): a match only counts as a HIT when the top unit beats the
        # runner-up by the margin. A near-tie means the query is ambiguous between adjacent units —
        # don't commit (and pollute that unit's seed); materialize the right unit instead.
        decisive = (best_score - second_score) >= self._hit_margin if second_score >= 0.0 else True
        if best_id is not None and best_score >= self._threshold and decisive:
            unit = self._units[best_id]
            self._refresh_if_expired(unit)
            if unit.is_fresh:
                # Behavioral seed: remember the query that hit (recording only in 0.3.0;
                # in-memory like the hit counter — not yet persisted per-hit).
                unit.touch(
                    query if self._learn_behavior else None,
                    max_queries=self._max_hit_queries,
                )
                cache_hit = True
            else:
                # Stale (dirtied by a change or TTL) -> re-materialize THIS unit.
                read_usage = self._materialize_into(unit, query, qe, ns)
                cache_hit = False
            confidence = best_score
        else:
            unit = self._new_unit(query, qe, ns)
            read_usage = self._materialize_into(unit, query, qe, ns)
            self._units[unit.id] = unit
            cache_hit = False
            confidence = max(best_score, 0.0)

        self._reads_total += 1
        if cache_hit:
            self._reads_hit += 1
            self._tokens_saved += unit.synth_tokens  # this read avoided rebuilding the unit

        evidence = list(unit.evidence)
        # ---- v0.4 ORDERED FLOW: recall first (free, from cache) -> re-cover -> escalate (RAG floor) ----
        # S1 — how well the SINGLE matched unit covers THIS query (cheap cosine over its claims).
        coverage = self._semantic_coverage(qe, unit)

        # Cross-unit recall (opt-in, semantic only): when the single unit under-covers, pool atomic
        # claims across ALL fresh units and let the BEST cached claim (from ANY unit) raise coverage —
        # so the escalation decision is made against what the WHOLE cache knows, not one unit. As the
        # cache fills, more queries are covered from cache => the escalation rate falls (compounding).
        recalled: list[RecalledClaim] = []
        if (
            self._cross_unit_recall
            and self._is_semantic_embedder()
            and coverage < self._effective_recall_threshold()
        ):
            recalled = self._recall_claims(qe, ns, limit=self._recall_limit)
            if recalled:
                coverage = max(coverage, max(claim.score for claim in recalled))  # S1c

        # S2 (opt-in) — the RELIABLE containment gate (cross-encoder / NLI / LLM entailment), consulted
        # ONLY in the ambiguous band [floor, ceiling) where cosine can't tell "adjacent" from "answers".
        # Judges what we would actually SERVE (unit + recalled). This is the gate Coalent OWNS.
        if (
            self._coverage_scorer is not None
            and self._coverage_floor <= coverage < self._coverage_ceiling
        ):
            coverage = self._coverage_scorer(query, self._with_recalled(unit.understanding, recalled))

        # Escalate to fresh raw (the RAG FLOOR) ONLY if the cache — the unit AND its cross-unit claims —
        # still under-covers: a retrieval, no LLM call (like plain RAG). Opt-in learn_on_escalation then
        # caches that fresh evidence as a NEW unit so the next similar query is a warm hit (compounding).
        escalated = False
        if (
            cache_hit
            and self._enable_coverage_escalation
            and coverage < self._coverage_floor
            and not self._emits_raw(strat, escalated=False)
            and self._emits_raw(strat, escalated=True)
            and not unit.understanding.get("_passthrough")
        ):
            fresh = self._retrieve(query, ns)
            evidence = self._augment(evidence, fresh)
            escalated = True
            self._reads_escalated += 1
            if self._learn_on_escalation:
                # An escalation implies a cache HIT, so read_usage is None here -> assign the
                # learned unit's synthesis cost (also rolled into stats() via _build_unit).
                read_usage = self._learn_from_escalation(query, qe, ns, fresh)

        # Project + inject the recalled claims (compact; raw spans only if recall_raw is opted in).
        # Serve: semantic-select the matched unit's atoms (select_floor) when enabled AND at least one
        # atom clears the floor; otherwise fall back to the v0.3 lexical projection (the conservative
        # path — never over-serves every atom just because nothing cleared a high floor).
        selected = (
            self._select_claim_texts(qe, unit)
            if self._select_floor is not None
            and self._is_semantic_embedder()
            and unit.claim_embeddings
            else []
        )
        if selected:
            served = dict(unit.understanding)
            served["claims"] = selected
            context = self._project(served, evidence, query, strat, escalated, trim_claims=False)
        else:
            context = self._project(unit.understanding, evidence, query, strat, escalated)
        if recalled:
            context = dict(context)
            understanding_view = dict(context.get("understanding", {}))
            understanding_view["recalled_claims"] = [claim.claim for claim in recalled]
            if self._recall_raw:  # OPT-IN only — default path stays compact (claims + summary)
                self._attach_recall_sources(query, recalled)
                spans: list[str] = []
                for claim in recalled:
                    if claim.source and claim.source not in spans:
                        spans.append(claim.source)
                if spans:
                    understanding_view["recalled_evidence"] = spans
            context["understanding"] = understanding_view

        # S3 affordance (we EXPOSE, don't own): the cache still under-covers even after recall + S2,
        # so the user's answerer may be ungrounded and they can retry / widen. `cache_hit` stays the
        # authoritative hit signal; this is only a "you may want fresh retrieval" hint.
        needs_retrieval = coverage < self._coverage_floor

        return Result(
            understanding=dict(unit.understanding),
            evidence=evidence,
            cache_hit=cache_hit,
            unit_id=unit.id,
            confidence=confidence,
            namespace=ns,
            related=self._related(unit, qe, related, ns),
            context=context,
            usage=read_usage,
            coverage=coverage,
            escalated=escalated,
            recalled=recalled,
            needs_retrieval=needs_retrieval,
        )

    def _match_score(self, qe: tuple[float, ...], unit: Cognition) -> float:
        """Hybrid match: key on what the unit KNOWS, with the seed query as a recall
        floor. ``topic`` = query<->understanding-embedding (kills surface-form false
        hits like "exchange policy" matching "leave policy"); ``seed`` = query<->seed
        query (keeps genuine paraphrases). Blend, weighted toward topic. Falls back to
        pure seed when the understanding embedding is missing (un-backfilled unit, or a
        zero vector under HashingEmbedder) so matching never silently under-fires."""
        seed = cosine(qe, unit.query_embedding)
        if self._route_by_claim and unit.claim_embeddings:
            # Late-interaction: route by the unit's BEST-matching claim, not its averaged
            # understanding — so a query finds the unit holding a claim about it, even in a
            # fat multi-claim unit, instead of landing on a topically-adjacent centroid.
            topic = max(cosine(qe, ce) for ce in unit.claim_embeddings)
        elif unit.understanding_embedding:
            topic = cosine(qe, unit.understanding_embedding)
        else:
            return seed
        w = self._understanding_weight
        return w * topic + (1.0 - w) * seed

    def _best_match(
        self, qe: tuple[float, ...], ns: str
    ) -> tuple[str | None, float, float]:
        """Best-matching unit in the namespace, plus the RUNNER-UP score so the caller can
        require a disambiguation margin (``hit_margin``) — the precision guard against a query
        being absorbed into a topically-adjacent neighbor when several units clear the bar."""
        best_id: str | None = None
        best = -1.0
        second = -1.0
        for unit_id, unit in self._units.items():
            if unit.namespace != ns:
                continue
            score = self._match_score(qe, unit)
            if score > best:
                best, second, best_id = score, best, unit_id
            elif score > second:
                second = score
        return best_id, best, second

    def _new_unit(self, query: str, qe: tuple[float, ...], ns: str) -> Cognition:
        key = hashlib.sha1(f"{ns}|{query}".encode("utf-8")).hexdigest()[:16]
        return Cognition(
            id=f"cog:{key}",
            namespace=ns,
            query=query,
            query_embedding=qe,
            understanding={},
            evidence=(),
            provenance=ProvenanceManifest("none", "none"),
        )

    def _retrieve(self, query: str, ns: str) -> list[Chunk]:
        """Retrieve, then optionally drop irrelevant chunks via the ``relevance_gate``
        hook (BYO reranker / score threshold). De-noises the understanding, provenance,
        AND the raw floor in one place — used by both materialize and escalation. With
        no gate it is plain retrieval. Coalent never reranks itself ("context != retriever")."""
        chunks = self._retriever.retrieve(query, namespace=ns or None)
        if self._relevance_gate is not None:
            chunks = list(self._relevance_gate(query, chunks))
        return chunks

    def _materialize_into(
        self, unit: Cognition, query: str, qe: tuple[float, ...], ns: str
    ) -> Usage | None:
        return self._build_unit(unit, query, qe, self._retrieve(query, ns))

    def _learn_from_escalation(
        self, query: str, qe: tuple[float, ...], ns: str, chunks: list[Chunk]
    ) -> Usage | None:
        """v0.4 (opt-in) — cache the escalation: synthesize the freshly-retrieved chunks into a NEW
        unit so the next similar query is a warm hit. The compounding driver — at the cost of one
        synthesis per escalation, which is why it is OFF by default."""
        unit = self._new_unit(query, qe, ns)
        usage = self._build_unit(unit, query, qe, chunks)
        self._units[unit.id] = unit
        return usage

    def _build_unit(
        self, unit: Cognition, query: str, qe: tuple[float, ...], chunks: list[Chunk]
    ) -> Usage | None:
        synthesis = self._synth.synthesize(query, chunks)
        self._account_usage(synthesis.usage)
        understanding = dict(synthesis.understanding)

        if not synthesis.ok:
            # Synthesis failed: never cache fabricated understanding. Keep the raw
            # evidence (the RAG floor) and conservatively depend on all sources.
            understanding["_synthesis_failed"] = True
            cited = list(chunks)
        else:
            cited = [chunks[i] for i in synthesis.used if 0 <= i < len(chunks)]
            if not cited:
                # No usable citations -> correctness over precision: depend on all
                # retrieved sources, and flag it (never silently widen unnoticed).
                cited = list(chunks)
                if chunks:
                    understanding["_citation_fallback"] = True

        spans = tuple(
            SourceSpan.from_text(chunk.artifact_id, chunk.text, version=chunk.version)
            for chunk in cited
        )
        # Seed query is the unit's BIRTH identity — set once (by _new_unit), never on
        # a stale re-materialize, so the key can't drift toward whatever triggered it.
        if not unit.query_embedding:
            unit.query = query
            unit.query_embedding = qe
        unit.understanding = understanding
        unit.evidence = tuple(chunks)  # retain ALL raw — the floor, regardless of citations
        unit.provenance = ProvenanceManifest("synth@1", "semantic@2", source_spans=spans)
        # Key on what the unit KNOWS: (re)compute the understanding + per-claim embeddings.
        unit.understanding_embedding, unit.claim_embeddings = self._cognition_embeddings(
            understanding
        )
        # Residual safety net (opt-in): retain number-bearing cited spans the extractor missed as
        # extra atoms, then RE-key from the augmented understanding so claim_embeddings stays exactly
        # parallel to _claim_texts (order = [claims..., residuals..., summary]). The recompute re-embeds
        # the original claims once — accepted: correctness of the parallelism invariant over saving a
        # single build-time embed on an opt-in path (a hit's serve/recall never re-embed).
        residual = self._residual_texts(unit.claim_embeddings, cited)
        if residual:
            existing = understanding.get("claims")
            base = list(existing) if isinstance(existing, list) else []
            understanding["claims"] = base + residual
            unit.understanding = understanding
            unit.understanding_embedding, unit.claim_embeddings = self._cognition_embeddings(
                understanding
            )
        unit.synth_tokens = synthesis.usage.total_tokens if synthesis.usage is not None else 0
        unit.touch()  # a build, not a hit -> no query recorded
        self._mark_fresh(unit)
        self._reindex(unit)
        self._persist(unit)
        return synthesis.usage

    def _account_usage(self, usage: Usage | None) -> None:
        """Roll a synthesis call's token cost into the cache totals (surfaced by ``stats()``)."""
        if usage is None:
            return
        self._synth_calls += 1
        self._synth_prompt_tokens += usage.prompt_tokens
        self._synth_completion_tokens += usage.completion_tokens
        self._synth_cost += usage.cost

    def _understanding_digest(self, understanding: dict[str, Any]) -> str:
        """Text the understanding-embedding is taken over: summary + claims + facts."""
        return self._text_of(understanding)

    @staticmethod
    def _claim_texts(understanding: dict[str, Any]) -> list[str]:
        """Atomic spans to embed for per-claim semantic coverage: each claim, plus the
        summary as a fallback so a summary-only unit still gets coverage. A structured
        claim (a dict, e.g. ``{"claim": ..., "source": ...}``) contributes its text field
        so per-claim embeddings stay clean rather than embedding dict syntax."""
        texts: list[str] = []
        claims = understanding.get("claims")
        if isinstance(claims, list):
            for claim in claims:
                if isinstance(claim, dict):
                    text = str(claim.get("claim") or claim.get("text") or claim).strip()
                else:
                    text = str(claim).strip()
                if text:
                    texts.append(text)
        summary = understanding.get("summary")
        if isinstance(summary, str) and summary.strip():
            texts.append(summary)
        return texts

    def _cognition_embeddings(
        self, understanding: dict[str, Any]
    ) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
        """Embed the understanding digest + each claim — the two embeddings the v0.3
        matcher/coverage key on. Per-claim is batched via ``embed_many`` when the
        embedder supports it. Centralized so materialize and backfill agree."""
        digest = self._understanding_digest(understanding)
        # Only key on the understanding when the digest has real lexical content; a
        # trivial digest embeds to a meaningless (often zero) vector that would drag
        # the blend down even for an identical query. Empty -> matcher uses pure seed.
        u_emb: tuple[float, ...] = tuple(self._embedder.embed(digest)) if tokenize(digest) else ()
        claim_texts = self._claim_texts(understanding)
        c_embs = tuple(tuple(v) for v in embed_texts(self._embedder, claim_texts))
        return u_emb, c_embs

    def _residual_texts(
        self, claim_embeddings: tuple[tuple[float, ...], ...], cited: list[Chunk]
    ) -> list[str]:
        """Residual safety net: number-bearing sentences from the CITED sources whose max cosine to
        the unit's atoms is below ``residual_floor`` — i.e. facts the extractor did not capture.
        Returned as verbatim texts to retain as extra atoms, ranked most-missed first and capped at
        ``residual_limit``. Embedding-only; no extra LLM call. Sentence splitting assumes standard
        end-of-sentence punctuation (``. ! ?`` + space); atypically-punctuated prose may fragment."""
        if self._residual_floor is None or not claim_embeddings:
            return []
        sentences = _number_sentences(cited)
        if not sentences:
            return []
        scored: list[tuple[float, str]] = []
        for text, raw_emb in zip(sentences, embed_texts(self._embedder, sentences)):
            emb = tuple(raw_emb)
            if not emb:
                continue
            covered = max((cosine(emb, ae) for ae in claim_embeddings), default=0.0)
            if covered < self._residual_floor:
                scored.append((covered, text))
        scored.sort(key=lambda cs: cs[0])  # lowest coverage (most clearly missed) first
        kept = scored if self._residual_limit <= 0 else scored[: self._residual_limit]
        return [text for _covered, text in kept]

    def _backfill_cognition(self, unit: Cognition) -> None:
        """One-time upgrade of a pre-v0.3 unit: compute + persist its embeddings."""
        unit.understanding_embedding, unit.claim_embeddings = self._cognition_embeddings(
            unit.understanding
        )
        self._persist(unit)

    # ------------------------------------------------ context intelligence
    def _semantic_coverage(self, qe: tuple[float, ...], unit: Cognition) -> float:
        """How well the unit's best single claim addresses THIS query: max cosine of the
        query against each per-claim embedding. 1.0 when there are no claims to judge by
        (can't prove a gap — don't penalize a structured/passthrough unit). Under
        HashingEmbedder this degrades to per-claim keyword overlap (a lexical floor);
        with a semantic embedder it catches paraphrased gaps a lexical gate would miss."""
        if not unit.claim_embeddings:
            return 1.0
        return max(cosine(qe, ce) for ce in unit.claim_embeddings)

    def _effective_recall_threshold(self) -> float:
        """The coverage below which cross-unit recall fires. A SEPARATE knob from the raw
        escalation gate (coverage_floor) so the two tune independently: recall is the
        cheap, retrieval-free first response to under-coverage; raw escalation is the
        fallback. Defaults to coverage_floor; sweep it on the eval for fewest misses."""
        return (
            self._recall_threshold
            if self._recall_threshold is not None
            else self._coverage_floor
        )

    def _is_semantic_embedder(self) -> bool:
        """Cross-unit claim recall needs MEANING, not keyword overlap — so it is gated to
        a real embedder. Under the zero-dep HashingEmbedder, claim cosine collapses to
        lexical overlap and recall would amplify keyword coincidence, so we skip it."""
        return not isinstance(self._embedder, HashingEmbedder)

    @staticmethod
    def _with_recalled(
        understanding: dict[str, Any], recalled: list[RecalledClaim]
    ) -> dict[str, Any]:
        """The understanding the S2 coverage_scorer should judge: the unit's, PLUS the cross-unit
        recalled claim texts — so containment is checked against what we will actually serve."""
        merged = dict(understanding)
        if recalled:
            existing = merged.get("claims")
            base = list(existing) if isinstance(existing, list) else []
            merged["claims"] = base + [claim.claim for claim in recalled]
        return merged

    def _select_claim_texts(self, qe: tuple[float, ...], unit: Cognition) -> list[str]:
        """Semantic-select serve: the matched unit's atoms whose per-claim embedding is within
        ``select_floor`` cosine of the query — serve the query-RELEVANT atoms by MEANING, not by
        keyword overlap. Excludes the summary (projected separately); de-duplicated, order kept."""
        summary = str(unit.understanding.get("summary", ""))
        floor = self._select_floor or 0.0
        out: list[str] = []
        seen: set[str] = set()
        for text, emb in zip(self._claim_texts(unit.understanding), unit.claim_embeddings):
            if not emb or not text or text == summary or text in seen:
                continue
            if cosine(qe, emb) >= floor:
                seen.add(text)
                out.append(text)
        return out

    def _recall_claims(
        self, qe: tuple[float, ...], ns: str, *, limit: int
    ) -> list[RecalledClaim]:
        """Cross-unit late interaction (v0.4 substrate). Pool the per-claim embeddings of
        EVERY fresh unit into one associative memory and rank atomic claims by MaxSim (max
        cosine to the query) pooled ACROSS units — so the answer-claim is recovered even
        when its unit's averaged understanding ranks below top-k. De-duplicated by text
        (highest score wins). This is the cross-document, multi-hop substrate that
        single-shot top-k retrieval cannot reach — at zero extra LLM call."""
        scored: list[RecalledClaim] = []
        for unit in self._units.values():
            if unit.namespace != ns or not unit.is_fresh:
                continue
            for text, emb in zip(self._claim_texts(unit.understanding), unit.claim_embeddings):
                if not emb or not text:
                    continue
                scored.append(RecalledClaim(claim=text, score=cosine(qe, emb), unit_id=unit.id))
        scored.sort(key=lambda r: r.score, reverse=True)
        out: list[RecalledClaim] = []
        seen: set[str] = set()
        for claim in scored:
            if claim.claim in seen:
                continue
            seen.add(claim.claim)
            out.append(claim)
            if len(out) >= limit:
                break
        return out

    def _attach_recall_sources(self, query: str, recalled: list[RecalledClaim]) -> None:
        """OPT-IN (``recall_raw``, default OFF) — ground each recalled claim in the retained raw
        span behind it (the unit evidence chunk most relevant to the query, by free lexical
        overlap). Recovers a fact a lossy claim dropped, but ~doubles tokens, so it is NOT the
        default — the compact claims+summary path is."""
        terms = set(tokenize(query))
        for claim in recalled:
            unit = self._units.get(claim.unit_id)
            if unit is None or not unit.evidence:
                continue
            claim.source = max(
                (chunk.text for chunk in unit.evidence),
                key=lambda text: len(terms & set(tokenize(text))),
                default="",
            )

    @staticmethod
    def _text_of(understanding: dict[str, Any]) -> str:
        """Flatten the understanding (summary + claims + entities + facts) to text — what
        the understanding-embedding and the digest are taken over."""
        parts: list[str] = []
        summary = understanding.get("summary")
        if isinstance(summary, str):
            parts.append(summary)
        for key in ("claims", "entities"):
            value = understanding.get(key)
            if isinstance(value, list):
                parts.extend(str(item) for item in value)
        facts = understanding.get("facts")
        if isinstance(facts, dict):
            parts.extend(f"{name} {val}" for name, val in facts.items())
        return " ".join(parts)

    @staticmethod
    def _augment(existing: list[Chunk], extra: list[Chunk]) -> list[Chunk]:
        seen = {(chunk.artifact_id, chunk.text) for chunk in existing}
        merged = list(existing)
        for chunk in extra:
            key = (chunk.artifact_id, chunk.text)
            if key not in seen:
                seen.add(key)
                merged.append(chunk)
        return merged

    @staticmethod
    def _emits_raw(strategy: str, escalated: bool) -> bool:
        """Whether the projected context will contain the raw evidence — the exact
        rule ``_project`` applies. The coverage gate MUST measure over the same
        payload, or the gate and the projector disagree (the RAG-floor bug)."""
        if strategy == ContextStrategy.CONTEXT_ONLY:
            return False
        if strategy == ContextStrategy.CONTEXT_RAW:
            return True
        return escalated  # CONTEXT_FIRST: raw only once escalated

    @staticmethod
    def _project(
        understanding: dict[str, Any],
        evidence: list[Chunk],
        query: str,
        strategy: str,
        escalated: bool,
        *,
        trim_claims: bool = True,
    ) -> dict[str, Any]:
        projected = SemanticCache._project_understanding(understanding, query, trim_claims=trim_claims)
        raw = (
            [chunk.text for chunk in evidence]
            if SemanticCache._emits_raw(strategy, escalated)
            else []
        )
        return {"understanding": projected, "raw": raw}

    @staticmethod
    def _project_understanding(
        understanding: dict[str, Any], query: str, *, trim_claims: bool = True
    ) -> dict[str, Any]:
        terms = set(tokenize(query))
        # Passthrough records (structured tool/API output) are already minimal — a
        # status/id/boolean with no lexical overlap is often THE decision-relevant
        # field, so never query-trim them; only LLM prose gets the relevance filter.
        # ``trim_claims=False`` also skips the lexical trim — used when claims were already
        # picked SEMANTICALLY upstream (select_floor), so we don't re-filter them by keywords.
        keep_all = bool(understanding.get("_passthrough")) or not trim_claims
        projected: dict[str, Any] = {}
        summary = understanding.get("summary")
        if summary is not None:
            projected["summary"] = summary
        claims = understanding.get("claims")
        if isinstance(claims, list):
            relevant = claims if keep_all else [c for c in claims if terms & set(tokenize(str(c)))]
            projected["claims"] = relevant or claims
        facts = understanding.get("facts")
        if isinstance(facts, dict):
            rel = facts if keep_all else {
                k: v for k, v in facts.items() if terms & set(tokenize(f"{k} {v}"))
            }
            projected["facts"] = rel or facts
        for flag in ("_synthesis_failed", "_citation_fallback", "_passthrough"):
            if understanding.get(flag):
                projected[flag] = understanding[flag]
        return projected

    # ------------------------------------------------------ agent affordances
    def drill(self, unit_id: str) -> list[Chunk]:
        """The 'drill into source' tool: full raw evidence behind a unit."""
        unit = self._units.get(unit_id)
        return list(unit.evidence) if unit is not None else []

    def widen(self, query: str, *, namespace: str | None = None) -> list[Chunk]:
        """The 'widen retrieval' tool: fetch fresh evidence for a query."""
        return self._retriever.retrieve(query, namespace=namespace)

    # ------------------------------------------- provenance + entity indexes
    def _reindex(self, unit: Cognition) -> None:
        for index in (self._artifact_index, self._entity_index):
            for key, unit_ids in list(index.items()):
                unit_ids.discard(unit.id)
                if not unit_ids:
                    del index[key]
        for artifact_id in unit.provenance.artifact_ids():
            self._artifact_index.setdefault(artifact_id, set()).add(unit.id)
        for entity in self._entities_of(unit):
            self._entity_index.setdefault(entity, set()).add(unit.id)

    @staticmethod
    def _entities_of(unit: Cognition) -> set[str]:
        raw = unit.understanding.get("entities", [])
        if not isinstance(raw, list):
            return set()
        return {str(item).strip().lower() for item in raw if str(item).strip()}

    # ----------------------------------------------- cross-unit relationships
    def _related(
        self, seed: Cognition, qe: tuple[float, ...], limit: int, ns: str
    ) -> list[Related]:
        """Up to ``limit`` units sharing an entity or source, ranked by query relevance."""
        if limit <= 0:
            return []
        candidates: dict[str, str] = {}
        for entity in self._entities_of(seed):
            for unit_id in self._entity_index.get(entity, ()):
                if unit_id != seed.id:
                    candidates.setdefault(unit_id, "shared_entity")
        for artifact_id in seed.provenance.artifact_ids():
            for unit_id in self._artifact_index.get(artifact_id, ()):
                if unit_id != seed.id:
                    candidates.setdefault(unit_id, "shared_source")

        scored: list[tuple[float, str, str, Cognition]] = []
        for unit_id, relation in candidates.items():
            unit = self._units.get(unit_id)
            if unit is None or not unit.is_fresh or unit.namespace != ns:
                continue
            scored.append((self._match_score(qe, unit), unit_id, relation, unit))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            Related(
                unit_id=unit_id,
                understanding=dict(unit.understanding),
                evidence=list(unit.evidence),
                relation=relation,
                score=score,
            )
            for score, unit_id, relation, unit in scored[:limit]
        ]

    # ----------------------------------------------------------- invalidation
    def invalidate(self, event: ChangeEvent) -> InvalidationResult:
        """Apply a change event. ``kind="delete"`` evicts; otherwise dirties only
        units whose provenance actually changed (skip no-ops). Warns when an event
        matches no cached units — the #1 wiring mistake (id mismatch)."""
        result = InvalidationResult()
        unit_ids = set(self._artifact_index.get(event.artifact_id, ()))
        result.matched_units = len(unit_ids)
        if not unit_ids:
            logger.warning(
                "change event for %r matched no cached units — does this id match the "
                "artifact_id your retrieval records as provenance?",
                event.artifact_id,
            )
            return result
        for unit_id in unit_ids:
            unit = self._units.get(unit_id)
            if unit is None:
                continue
            if event.kind == "delete":
                self._evict(unit_id)
                result.deleted.append(unit_id)
            elif self._content_changed(unit, event):
                unit.mark_dirty()
                self._persist(unit)
                result.dirtied.append(unit_id)
            else:
                result.skipped_unchanged.append(unit_id)
        return result

    def source_changed(
        self, artifact_id: str, *, text: str | None = None, version: str | None = None
    ) -> InvalidationResult:
        """Convenience: hash the new content (if given) and fire a change event."""
        content_hash = ""
        if text is not None:
            content_hash = SourceSpan.from_text(artifact_id, text).content_hash
        return self.invalidate(
            ChangeEvent(artifact_id=artifact_id, version=version or "", content_hash=content_hash)
        )

    def source_deleted(self, artifact_id: str) -> InvalidationResult:
        """A source was removed: evict the units that depended on it."""
        return self.invalidate(ChangeEvent(artifact_id=artifact_id, kind="delete"))

    def _evict(self, unit_id: str) -> None:
        self._units.pop(unit_id, None)
        for index in (self._artifact_index, self._entity_index):
            for key, unit_ids in list(index.items()):
                unit_ids.discard(unit_id)
                if not unit_ids:
                    del index[key]
        if self._store is not None:
            self._store.delete(unit_id)

    # --------------------------------------------------- time-based freshness
    def _refresh_if_expired(self, unit: Cognition) -> None:
        """On TTL expiry, revalidate feed-less sources by hash; skip no-op rebuilds."""
        policy = self._freshness
        if policy is None or policy.max_age is None or not unit.is_fresh:
            return
        if self._clock() - unit.freshness_epoch <= policy.max_age:
            return
        if policy.revalidate is None:
            unit.mark_dirty()  # no revalidator -> conservatively rebuild on read
            self._persist(unit)
            return
        changed = False
        for artifact_id in unit.provenance.artifact_ids():
            try:
                fetched = policy.revalidate(artifact_id)
            except Exception:  # revalidation must never crash a read
                fetched = None
            if fetched is None:
                continue
            text, version = fetched
            event = ChangeEvent(
                artifact_id=artifact_id,
                version=version,
                content_hash=SourceSpan.from_text(artifact_id, text).content_hash,
            )
            if self._content_changed(unit, event):
                changed = True
                break
        if changed:
            unit.mark_dirty()
        else:
            self._mark_fresh(unit)  # content unchanged -> bump freshness, no rebuild
        self._persist(unit)

    @staticmethod
    def _content_changed(unit: Cognition, event: ChangeEvent) -> bool:
        spans = unit.provenance.spans_for(event.artifact_id)
        if not spans:
            return True
        return any(SemanticCache._span_differs(event, span) for span in spans)

    @staticmethod
    def _span_differs(event: ChangeEvent, span: SourceSpan) -> bool:
        if event.content_hash and span.content_hash:
            return event.content_hash != span.content_hash
        if event.version and span.version:
            return event.version != span.version
        return True  # cannot prove unchanged -> conservatively changed

    # ------------------------------------------------------------------ misc
    @staticmethod
    def cost_sensitive_floor(escalation_cost: float, miss_cost: float) -> float:
        """v0.4 (M3) — derive ``coverage_floor`` from a COST trade-off instead of guessing a
        constant: ``tau* = 1 - escalation_cost / miss_cost``. The dearer a WRONG answer
        (``miss_cost``) is relative to one extra retrieval (``escalation_cost``), the higher the
        floor → the more eagerly the cache escalates to the RAG floor. Pass the result as
        ``coverage_floor=``. Clamped to [0, 1]; a non-positive ``miss_cost`` yields 0 (never escalate)."""
        if miss_cost <= 0:
            return 0.0
        return round(max(0.0, min(1.0, 1.0 - escalation_cost / miss_cost)), 4)

    def stats(self) -> dict[str, float | int | None]:
        """Cache size + read-time observability. ``escalation_rate`` is the fraction of
        cache HITS that fell back to fresh raw — a high value means the cached
        understanding under-covers real queries (deepen it, or lower coverage_floor)."""
        hits = self._reads_hit
        synth_tokens = self._synth_prompt_tokens + self._synth_completion_tokens
        avg_synth = synth_tokens / self._synth_calls if self._synth_calls else 0.0
        return {
            "units": len(self._units),
            "tracked_artifacts": len(self._artifact_index),
            "reads": self._reads_total,
            "hits": hits,
            "escalations": self._reads_escalated,
            "escalation_rate": round(self._reads_escalated / hits, 3) if hits else 0.0,
            "hit_rate": round(hits / self._reads_total, 3) if self._reads_total else 0.0,
            # Token/cost telemetry. synth_tokens = total spent building units. tokens_saved =
            # EXACT sum of each hit's unit build cost (per-unit recorded, so it survives a store
            # reload and is correct under mixed/usage-less providers — not a running-average guess).
            "synth_calls": self._synth_calls,
            "synth_prompt_tokens": self._synth_prompt_tokens,
            "synth_completion_tokens": self._synth_completion_tokens,
            "synth_tokens": synth_tokens,
            "synth_cost": round(self._synth_cost, 6),
            "avg_synth_tokens": round(avg_synth, 1),
            "tokens_saved": self._tokens_saved,
            # Effective operating thresholds/knobs, so the cost/risk point is observable (M3 tau*)
            # and a caller can confirm which additive mechanisms are actually active this run.
            "coverage_floor": self._coverage_floor,
            "recall_threshold": self._effective_recall_threshold(),
            "hit_threshold": self._threshold,
            "hit_margin": self._hit_margin,
            "select_floor": self._select_floor,
            "residual_floor": self._residual_floor,
        }
