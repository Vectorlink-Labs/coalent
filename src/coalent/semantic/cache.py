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

from ..domain.models import ChangeEvent, ProvenanceManifest, SourceSpan, Status
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

try:  # optional acceleration for pool serving at scale (``pip install coalent[fast]``).
    # The core stays zero-dependency: every numpy path has a pure-Python twin.
    import numpy as _np
except ImportError:  # pragma: no cover - environment-dependent
    _np = None  # type: ignore[assignment]

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


# Workload presets (v0.5): named bundles of TUNED knob values — the operating points our
# benchmarks validated — applied only to knobs the caller left unset (explicit kwargs win).
# "multi_hop" is the bench_multihop fewest-misses point: recall_threshold 0.70 (dormant at the
# coverage_floor default) + the bridge restart, so cross-document questions work out of the box.
PRESETS: dict[str, dict[str, float | bool]] = {
    "default": {},
    "multi_hop": {"recall_threshold": 0.7, "recall_bridge": True},
}


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
        preset: str | None = None,
        embedder: Embedder | None = None,
        hit_threshold: float | None = None,
        adaptive_hit: bool = False,
        reuse_threshold: float = 0.9,
        provenance_admission: bool = False,
        widen_chunks: int | None = None,
        source_fetcher: Callable[[str], list[Chunk]] | None = None,
        widen_on_admission: bool | None = None,
        split_by_artifact: bool = False,
        serve: str = "unit",
        serve_budget: int = 600,
        pool_header: Callable[[Cognition], str] | None = None,
        fast: bool | str = "auto",
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
        recall_bridge: bool | None = None,   # None = off unless a preset turns it on
        bridge_limit: int = 3,
        select_floor: float | None = None,
        residual_floor: float | None = None,
        residual_limit: int = 24,
        strategy: str = ContextStrategy.CONTEXT_FIRST,
        store: CognitionStore | None = None,
        freshness: FreshnessPolicy | None = None,
        clock: Callable[[], float] = time.time,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        # Workload preset (v0.5): a named bundle of tuned knob values, applied ONLY to knobs the
        # caller left unset — an explicit kwarg always wins. "default" is a no-op; "multi_hop"
        # arms the validated bridge configuration (recall_threshold 0.7 + recall_bridge) so the
        # cross-document capability fires out of the box instead of staying dormant.
        if preset is not None:
            if preset not in PRESETS:
                raise ValueError(
                    f"unknown preset {preset!r}; available: {', '.join(sorted(PRESETS))}"
                )
            overlay = PRESETS[preset]
            if recall_threshold is None and "recall_threshold" in overlay:
                recall_threshold = float(overlay["recall_threshold"])
            if recall_bridge is None and "recall_bridge" in overlay:
                recall_bridge = bool(overlay["recall_bridge"])
        self._retriever = retriever
        self._synth = synthesizer
        self._embedder: Embedder = embedder if embedder is not None else default_embedder()
        # Thresholds derive from the embedder when unset: OpenAI cosines are compressed
        # (~0.33), the lexical HashingEmbedder scores higher (~0.6) — one fixed default
        # can't fit both. Override explicitly, or tune via coalent.calibrate_thresholds.
        hit_default, cov_default = default_thresholds_for(self._embedder)
        self._threshold = hit_default if hit_threshold is None else hit_threshold
        # v0.5 — adaptive hit gate (M4 warm-up finding: match scores INFLATE as units
        # accumulate, so a fixed threshold sinks below the noise floor and the cache
        # 'absorbs' everything, including unanswerable queries). Self-calibrates against
        # the cache's own cross-unit score distribution; opt-in.
        self._adaptive_hit = adaptive_hit
        self._noise_ceiling = 0.0
        self._builds_since_calib = 0
        # v0.5 gate v2 — the REUSE channel: a query whose SEED similarity to a cached unit is
        # near-identity is the same question asked again; it must hit regardless of the noise
        # bar (M4 replay finding: the adaptive bar killed revisits; the blend diluted the seed
        # signal 70/30 — this reads the seed channel directly).
        self._reuse_threshold = reuse_threshold
        # v0.5 gate v3 — ADMISSION BY PROVENANCE (the sick-leave-paraphrase fix): before
        # building on a low-score read, probe retrieval; if every retrieved source is already
        # understood by cached units, building would mint a DUPLICATE understanding (the
        # GraphRAG tax) — serve the best covering unit instead. Exact, immune to score drift.
        self._provenance_admission = provenance_admission
        # v0.5 — BUILD-TIME SOURCE WIDENING (the keyhole fix): when a build is already
        # triggered by a live miss, fetch up to ``widen_chunks`` chunks OF that source
        # (via ``source_fetcher`` or a duck-typed ``retriever.widen``) instead of settling
        # for the few chunks the query surfaced. Replay-measured: keyhole units read a
        # median 1 chunk of ~24 (answer-presence 0.32 vs 0.62 widened). Lazy covenant
        # intact: widening NEVER fires at ingest, only inside a miss-triggered build.
        self._widen_chunks = widen_chunks
        self._source_fetcher = source_fetcher
        self._widen_on_admission = (widen_on_admission if widen_on_admission is not None
                                    else widen_chunks is not None)
        self._widen_warned = False
        # v0.5 EXPERIMENTAL preview of the v0.6 pool-first read path: serve="pool" swaps ONLY
        # the served payload — the global fresh-claim pool ranked against the query, packed to
        # ``serve_budget`` tokens, grouped per source unit (``pool_header`` prepends a caller-
        # supplied title line per group). Hit/build/admission/widening/freshness are untouched.
        # Measured basis (held-out n=605, pre-registered): pool 0.699@1036 vs the unit-anchored
        # path 0.579@706 (McNemar z=6.66); ties naive-k9 at 0.79x its tokens (z=0.60).
        if serve not in ("unit", "pool"):
            raise ValueError(f"serve must be 'unit' or 'pool', got {serve!r}")
        self._serve = serve
        self._serve_budget = serve_budget
        self._pool_header = pool_header
        self._pool_state: tuple[Any, ...] | None = None   # lazy (marker, texts, unit_ids, embs)
        # v0.5 — fast="auto": when numpy is importable the three O(cache-size) scans
        # (_best_match / _recall_claims / _bridge_claims) run their vectorized twins —
        # SAME control flow, matrix math instead of pure-Python cosines (equivalence is
        # pinned by tests/test_v05_fastpath.py). "auto" = numpy present; True without
        # numpy warns and falls back. The core stays zero-dependency either way.
        if fast == "auto":
            self._fast_enabled = _np is not None
        else:
            self._fast_enabled = bool(fast) and _np is not None
            if fast is True and _np is None:
                logger.warning("fast=True requested but numpy is not installed "
                               "(pip install coalent[fast]) — using the pure-Python path")
        self._fast_state: tuple[Any, ...] | None = None
        # v0.5 — per-source builds (the multi-source retrieval fix): when retrieval mixes
        # chunks from several artifacts, synthesize ONE UNIT PER ARTIFACT instead of one
        # blended unit that keeps the dominant topic and drops the rest. Opt-in.
        self._split_by_artifact = split_by_artifact
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
        # v0.5 — bridge restart (the bench_multihop "(d)" arm, promoted into the library): after
        # cross-unit recall fires, expand from the BRIDGE entity the matched unit names — serving
        # extras only, never a coverage signal. Off unless set or armed by the multi_hop preset.
        self._recall_bridge = bool(recall_bridge) if recall_bridge is not None else False
        self._bridge_limit = bridge_limit
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
        # v0.5 — freshness observability: a lightweight structured-event hook + counters, so
        # "what did the AI know, and when was it invalidated" is visible, not just enforced.
        # The hook must NEVER break serving: exceptions are swallowed (logged at debug).
        self._on_event = on_event
        self._stale_reads_prevented = 0    # reads that would have served STALE knowledge
        self._invalidated_units = 0        # units dirtied/evicted by change events
        self._age_serve_sum = 0.0          # age-at-serve accumulators (fresh hits only)
        self._age_serve_max = 0.0
        self._age_serve_n = 0
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
        is_reuse = False
        if best_id is not None and self._adaptive_hit:
            cand = self._units[best_id]
            if cand.query_embedding and cosine(qe, cand.query_embedding) >= self._reuse_threshold:
                is_reuse = True                       # same question, cached before: always a hit
        if (
            self._provenance_admission
            and not is_reuse
            and (best_id is None or best_score < self._effective_hit_threshold() or not decisive)
        ):
            probe = self._retrieve(query, ns)
            probe_ids = {c.artifact_id for c in probe if c.artifact_id}
            all_contained = bool(probe) and all(self._chunk_contained(c, ns) for c in probe)
            if not all_contained and self._widen_on_admission and probe_ids:
                # covered-but-thin sources: dirty the covering unit and ROUTE THE READ TO IT —
                # the hit path's stale branch then widen-rebuilds it IN PLACE (same unit id,
                # no duplicate). Self-extinguishing: fires ~once per source (96%->6% with
                # widening on). Genuinely novel sources still fall through to MISS.
                dirtied: set[str] = set()
                for a in probe_ids:
                    for uid in self._artifact_index.get(a, ()):
                        u_ = self._units.get(uid)
                        if u_ is not None and u_.is_fresh and not all(
                            self._chunk_contained(c, ns) for c in probe if c.artifact_id == a
                        ):
                            u_.mark_dirty()
                            dirtied.add(uid)
                if dirtied and all(a in self._artifact_index for a in probe_ids):
                    best_id = max(dirtied, key=lambda uid: self._match_score(qe, self._units[uid]))
                    best_score = self._match_score(qe, self._units[best_id])
                    is_reuse = True
                    self._emit("admission_widen_rebuild", unit_id=best_id,
                               probed_sources=len(probe_ids))
            if all_contained:
                covering: set[str] = set()
                for a in probe_ids:
                    covering |= self._artifact_index[a]
                covering = {u for u in covering
                            if u in self._units and self._units[u].namespace == ns
                            and self._units[u].is_fresh}
                if covering:
                    best_id = max(covering, key=lambda uid: self._match_score(qe, self._units[uid]))
                    best_score = self._match_score(qe, self._units[best_id])
                    is_reuse = True                   # duplicate build averted: hit-by-provenance
                    self._emit("admission_reuse", unit_id=best_id, probed_sources=len(probe_ids))
        if best_id is not None and (
            is_reuse or (best_score >= self._effective_hit_threshold() and decisive)
        ):
            unit = self._units[best_id]
            self._refresh_if_expired(unit)
            # Self-heal (v0.5): a unit whose synthesis FAILED must never be served hollow —
            # treat the hit as stale and re-materialize now (the lazy retry). Discovered in
            # M4: transient build failures were being cached and served as empty context.
            if unit.is_fresh and unit.understanding.get("_synthesis_failed"):
                unit.mark_dirty()
            if unit.is_fresh:
                # Behavioral seed: remember the query that hit (recording only in 0.3.0;
                # in-memory like the hit counter — not yet persisted per-hit).
                unit.touch(
                    query if self._learn_behavior else None,
                    max_queries=self._max_hit_queries,
                )
                cache_hit = True
                age = max(self._clock() - unit.freshness_epoch, 0.0)
                self._age_serve_sum += age
                self._age_serve_max = max(self._age_serve_max, age)
                self._age_serve_n += 1
            else:
                # Stale (dirtied by a change or TTL) -> re-materialize THIS unit. This is THE
                # freshness moment: without the rebuild, this read would have served stale
                # knowledge — counted and emitted so the prevention is visible, not silent.
                self._stale_reads_prevented += 1
                self._emit(
                    "stale_read_prevented",
                    unit_id=unit.id,
                    query=query,
                    unit_age_s=round(max(self._clock() - unit.freshness_epoch, 0.0), 3),
                )
                read_usage = self._materialize_into(unit, query, qe, ns)
                cache_hit = False
                self._emit("unit_rebuilt", unit_id=unit.id, reason="stale")
            confidence = best_score
        else:
            unit = self._new_unit(query, qe, ns)
            read_usage = self._materialize_into(unit, query, qe, ns)
            self._units[unit.id] = unit
            cache_hit = False
            confidence = max(best_score, 0.0)
            self._emit("unit_built", unit_id=unit.id, reason="miss")

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
            pre_recall_coverage = coverage
            recalled = self._recall_claims(qe, ns, limit=self._recall_limit)
            if recalled:
                coverage = max(coverage, max(claim.score for claim in recalled))  # S1c
                self._emit(
                    "claims_recalled",
                    unit_id=unit.id,
                    n=len(recalled),
                    pre_coverage=round(pre_recall_coverage, 4),
                    post_coverage=round(coverage, 4),
                )
                if self._recall_bridge:
                    # Bridge restart (v0.5, multi_hop preset): hop-2 resembles the BRIDGE entity
                    # the matched unit names, not the question — so rank OTHER units' claims by
                    # similarity to the matched unit's OWN claims and serve the best of them.
                    # Serving extras ONLY: bridge scores measure bridge-similarity, so they are
                    # deliberately kept OUT of `coverage` — the escalation gate stays honest
                    # about what actually answers THIS query.
                    bridged = self._bridge_claims(unit, recalled, ns)
                    if bridged:
                        self._emit(
                            "bridge_claims",
                            unit_id=unit.id,
                            n=len(bridged),
                            units=sorted({rc.unit_id for rc in bridged}),
                        )
                    recalled = recalled + bridged

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

        if self._serve == "pool":
            # v0.5 preview: the decision machinery above ran unchanged; only the served
            # payload becomes the global fresh-claim pool. Renderers read context["pool"].
            pool_text = self._pool_context(qe, ns)
            if pool_text:
                context = dict(context)
                context["pool"] = pool_text
                context["serve"] = "pool"

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

    def _effective_hit_threshold(self) -> float:
        """The hit bar actually applied. With ``adaptive_hit``, never below the cache's own
        cross-unit noise ceiling — the p95 of best-match scores among units KNOWN to be about
        different things. As the cache grows and scores inflate, the bar rises with them, so
        a falling build rate stays a REAL signal (M4 Grid-C finding)."""
        if not self._adaptive_hit:
            return self._threshold
        if self._noise_ceiling == 0.0 or self._builds_since_calib >= 16:
            self._noise_ceiling = self._noise_floor()
            self._builds_since_calib = 0
        return max(self._threshold, self._noise_ceiling + 0.02)

    def _noise_floor(self) -> float:
        """p95 of cross-unit best-match scores over a fixed-seed sample of the cache's own
        units: how high the blend runs on content that is about SOMETHING ELSE. Sampled and
        amortized (recomputed every 16 builds), embedding-only."""
        import random as _random

        units = [u for u in self._units.values() if u.query_embedding]
        if len(units) < 8:
            return 0.0
        sample = _random.Random(len(units)).sample(units, min(24, len(units)))
        bests: list[float] = []
        for probe in sample:
            best = 0.0
            for other in self._units.values():
                if other.id == probe.id or other.namespace != probe.namespace:
                    continue
                best = max(best, self._match_score(probe.query_embedding, other))
            bests.append(best)
        bests.sort()
        return bests[int(0.95 * (len(bests) - 1))]

    # ------------------------------------------------ fast scan twin (v0.5, optional numpy)
    def _fast_index(self) -> tuple[Any, ...] | None:
        """Lazy q-independent structures for the vectorized scans; invalidated by the same
        marker as the pool (unit added/removed/freshness flip). None when numpy is absent."""
        if not self._fast_enabled or _np is None:
            return None
        marker = self._pool_marker()
        state = self._fast_state
        if state is not None and state[0] == marker:
            return state
        uids = list(self._units.keys())
        dim = 0
        for u in self._units.values():
            for e in (u.query_embedding, u.understanding_embedding,
                      *(u.claim_embeddings or ())):
                if e:
                    dim = len(e)
                    break
            if dim:
                break
        dim = dim or 1
        def _norm(m: Any) -> Any:
            n = _np.linalg.norm(m, axis=1, keepdims=True)
            return m / _np.where(n > 0, n, 1.0)
        SE = _np.zeros((len(uids), dim))
        UE = _np.zeros((len(uids), dim))
        spans: list[tuple[int, int]] = []            # per-unit [start, end) into the claim rows
        has_empty: list[bool] = []                   # unit has a truthy-but-empty claim entry
        c_rows: list[tuple[float, ...]] = []
        c_meta: list[tuple[str, str, bool, bool, str]] = []  # (text, uid, atomic, fresh, ns)
        for i, (uid, u) in enumerate(self._units.items()):
            if u.query_embedding and len(u.query_embedding) == dim:
                SE[i] = u.query_embedding
            if u.understanding_embedding and len(u.understanding_embedding) == dim:
                UE[i] = u.understanding_embedding
            start = len(c_rows)
            empty = False
            atomic = {t for t in u.understanding.get("claims") or [] if isinstance(t, str)}
            for text, emb in zip(self._claim_texts(u.understanding), u.claim_embeddings or ()):
                if not emb:
                    empty = True
                    continue
                c_rows.append(emb)
                c_meta.append((str(text), uid, str(text) in atomic, u.is_fresh, u.namespace))
            spans.append((start, len(c_rows)))
            has_empty.append(empty)
        CE = _norm(_np.asarray(c_rows)) if c_rows else _np.zeros((0, dim))
        state = (marker, uids, _norm(SE), _norm(UE), spans, has_empty, CE, c_meta, dim)
        self._fast_state = state
        return state

    def _fast_scores(self, idx: tuple[Any, ...], qe: tuple[float, ...]) -> tuple[Any, Any, Any]:
        """(seed_sims, ue_sims, claim_sims) for one query — the only heavy math, vectorized."""
        _, _, SE, UE, _, _, CE, _, dim = idx
        q = _np.asarray(qe if len(qe) == dim else (0.0,) * dim, dtype=_np.float64)
        n = _np.linalg.norm(q)
        q = q / (n if n > 0 else 1.0)
        return SE @ q, UE @ q, (CE @ q if CE.shape[0] else _np.zeros(0))

    def _match_score_fast(self, idx: tuple[Any, ...], i: int, unit: Cognition,
                          seed_sims: Any, ue_sims: Any, claim_sims: Any) -> float:
        """Vectorized twin of ``_match_score`` — identical branch structure."""
        seed = float(seed_sims[i]) if unit.query_embedding else 0.0
        if self._route_by_claim and unit.claim_embeddings:
            start, end = idx[4][i]
            best = float(claim_sims[start:end].max()) if end > start else None
            if idx[5][i]:                            # empty entries score 0.0 in the pure max
                best = max(best, 0.0) if best is not None else 0.0
            topic = best if best is not None else 0.0
        elif unit.understanding_embedding:
            topic = float(ue_sims[i])
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
        idx = self._fast_index()
        best_id: str | None = None
        best = -1.0
        second = -1.0
        if idx is not None:
            seed_sims, ue_sims, claim_sims = self._fast_scores(idx, qe)
            for i, (unit_id, unit) in enumerate(self._units.items()):
                if unit.namespace != ns:
                    continue
                score = self._match_score_fast(idx, i, unit, seed_sims, ue_sims, claim_sims)
                if score > best:
                    best, second, best_id = score, best, unit_id
                elif score > second:
                    second = score
            return best_id, best, second
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

    def _widen_group(self, artifact_id: str, trigger: list[Chunk]) -> list[Chunk]:
        """Fetch the SOURCE's chunks (not the query's keyhole view) for a build already in
        flight. Merge = fetched (document order) with trigger chunks guaranteed present,
        truncated to ``widen_chunks``. Falls back to the trigger chunks (today's behavior)
        when no widening capability exists — never crashes, never silently substitutes."""
        if not self._widen_chunks:
            return trigger
        fetched: list[Chunk] = []
        if self._source_fetcher is not None:
            fetched = list(self._source_fetcher(artifact_id))
        else:
            widen = getattr(self._retriever, "widen", None)
            if callable(widen):
                fetched = list(widen(artifact_id, limit=self._widen_chunks))
            elif not self._widen_warned:
                self._widen_warned = True
                logger.warning("widen_chunks set but retriever has no widen() and no "
                               "source_fetcher given — building keyhole units")
                self._emit("widen_unavailable", artifact_id=artifact_id)
        if not fetched:
            return trigger
        seen = {c.text for c in fetched}
        merged = list(fetched) + [c for c in trigger if c.text not in seen]
        if len(merged) > self._widen_chunks:
            head = merged[: self._widen_chunks]
            missing = [c for c in trigger if c.text not in {x.text for x in head}]
            merged = (head[: self._widen_chunks - len(missing)] + missing
                      if missing else head)
        return merged

    def _chunk_contained(self, chunk: Chunk, ns: str = "") -> bool:
        """CONTAINMENT predicate (v0.5 admission fix): artifact coverage is only honest if
        this exact chunk's text is retained by a fresh covering unit. The boolean
        artifact-index check was provenance-DISHONEST with keyhole units (96% of admission
        hits pointed at units that had never read the probed chunk)."""
        for uid in self._artifact_index.get(chunk.artifact_id, ()):
            unit = self._units.get(uid)
            if unit is not None and unit.is_fresh and unit.namespace == ns:
                if any(ev.text == chunk.text for ev in unit.evidence):
                    return True
        return False

    def _materialize_into(
        self, unit: Cognition, query: str, qe: tuple[float, ...], ns: str
    ) -> Usage | None:
        chunks = self._retrieve(query, ns)
        if not self._split_by_artifact:
            if self._widen_chunks and chunks:
                counts: dict[str, int] = {}
                for c in chunks:
                    counts[c.artifact_id] = counts.get(c.artifact_id, 0) + 1
                dom = max(counts, key=lambda k: counts[k])
                dom_chunks = [c for c in chunks if c.artifact_id == dom]
                return self._build_unit(unit, query, qe, self._widen_group(dom, dom_chunks))
            return self._build_unit(unit, query, qe, chunks)
        groups: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            groups.setdefault(chunk.artifact_id, []).append(chunk)
        if len(groups) <= 1:
            only = next(iter(groups), None)
            built = (self._widen_group(only, chunks)
                     if only is not None and self._widen_chunks else chunks)
            return self._build_unit(unit, query, qe, built)
        # One unit per SOURCE: the dominant artifact keeps this unit's identity; every other
        # artifact gets its own sibling unit — so a mixed retrieval can never blend topics
        # into one lossy understanding (the M4 digest/multi-source finding).
        ordered = sorted(groups.items(), key=lambda kv: -len(kv[1]))
        # never re-synthesize an already-CONTAINED source; a covered-but-thin source gets its
        # EXISTING unit rebuilt widened (no duplicate) — the v0.5 containment semantics
        def _group_contained(chs: list[Chunk]) -> bool:
            return all(self._chunk_contained(c, ns) for c in chs)

        todo: list[tuple[str, list[Chunk], Cognition | None]] = []
        for artifact_id, chs in ordered:
            if artifact_id in self._artifact_index:
                if _group_contained(chs):
                    continue                          # honestly covered: skip
                existing = next((self._units[u] for u in self._artifact_index[artifact_id]
                                 if u in self._units and self._units[u].is_fresh), None)
                todo.append((artifact_id, chs, existing))
            else:
                todo.append((artifact_id, chs, None))
        if not todo:
            todo = [(ordered[0][0], ordered[0][1], None)]
        usage: Usage | None = None
        first = True
        for artifact_id, chs, existing in todo:
            built_chunks = self._widen_group(artifact_id, chs)
            if existing is not None:                  # widen-rebuild in place: no duplicate
                self._build_unit(existing, query, qe, built_chunks)
                self._emit("unit_rebuilt", unit_id=existing.id, reason="widen",
                           artifact_id=artifact_id)
                if first:
                    first = False
                continue
            if first:
                usage = self._build_unit(unit, query, qe, built_chunks)
                first = False
                continue
            sibling = self._new_unit(f"{query} · {artifact_id}", qe, ns)
            self._build_unit(sibling, query, qe, built_chunks)
            self._units[sibling.id] = sibling
            self._emit("unit_built", unit_id=sibling.id, reason="split",
                       artifact_id=artifact_id)
        if usage is None:                             # dominant was a widen-rebuild
            usage = self._build_unit(unit, query, qe,
                                     self._widen_group(todo[0][0], todo[0][1]))
        return usage

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
        self._builds_since_calib += 1
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
        query against each per-claim embedding. With no claim embeddings the verdict
        depends on whether there is anything SERVABLE at all: a structured/passthrough
        unit with real content keeps the benign 1.0 ("can't prove a gap — don't
        penalize"), but an EMPTY unit (failed synthesis: no claims, no summary) reports
        0.0 — v0.5 fix for the M4-discovered poison where hollow units claimed perfect
        coverage, suppressing recall AND the RAG-floor escalation, and served nothing."""
        if not unit.claim_embeddings:
            return 1.0 if self._text_of(unit.understanding).strip() else 0.0
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

    # ------------------------------------------------ pool serving (v0.5 preview of v0.6)
    def _pool_marker(self) -> tuple[int, int]:
        """Cheap invalidation key for the lazy pool: changes whenever a unit is added,
        removed, or flips freshness — the three events that alter the servable pool."""
        h = 0
        for uid, u in self._units.items():
            h ^= hash((uid, u.status is Status.FRESH))
        return (len(self._units), h)

    def _pool_rows(self, ns: str) -> tuple[list[str], list[str], list[tuple[float, ...]]]:
        """(claim_text, owner_unit_id, embedding) rows over FRESH units in ``ns`` only —
        freshness AND namespace isolation are the pool's contract: a stale unit's claims
        never serve, and a namespace never sees another namespace's claims (0.5.1 fix)."""
        texts: list[str] = []
        owners: list[str] = []
        embs: list[tuple[float, ...]] = []
        for uid, u in self._units.items():
            if u.status is not Status.FRESH or u.namespace != ns:
                continue
            claims = u.understanding.get("claims")
            if not isinstance(claims, list):
                continue
            ce = (u.claim_embeddings or ())[: len(claims)]
            for t, e in zip(claims, ce):
                s = (str(t.get("claim") or t.get("text") or t) if isinstance(t, dict)
                     else str(t)).strip()
                if s and e:
                    texts.append(s)
                    owners.append(uid)
                    embs.append(tuple(e))
        return texts, owners, embs

    def _pool_context(self, qe: tuple[float, ...], ns: str = "") -> str:
        """Global-pool serving: rank every fresh claim against the query, pack to
        ``serve_budget`` (~4 chars/token estimate), group consecutive picks under their
        owner unit with an optional caller-supplied header line. numpy when available;
        the pure-Python twin is exact but O(pool) — fine at small caches."""
        marker = (ns, *self._pool_marker())
        state = self._pool_state
        if state is None or state[0] != marker:
            texts, owners, embs = self._pool_rows(ns)
            matrix = None
            if _np is not None and embs:
                matrix = _np.asarray(embs, dtype=_np.float64)
                matrix /= _np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
            state = (marker, texts, owners, embs, matrix)
            self._pool_state = state
        _, texts, owners, embs, matrix = state
        if not texts:
            return ""
        if matrix is not None:
            q = _np.asarray(qe, dtype=_np.float64)
            q /= _np.linalg.norm(q) + 1e-9
            order = [int(i) for i in _np.argsort(-(matrix @ q), kind="stable")]
        else:
            order = sorted(range(len(texts)), key=lambda i: -cosine(qe, embs[i]))
        used = 0
        picked: list[int] = []
        for i in order:
            used += max(1, len(texts[i]) // 4)
            picked.append(i)
            if used >= self._serve_budget:
                break
        groups: list[tuple[str, list[str]]] = []
        cur_owner: str | None = None
        cur: list[str] = []
        for i in picked:
            if owners[i] != cur_owner and cur:
                groups.append((cur_owner or "", cur))
                cur = []
            cur_owner = owners[i]
            cur.append(texts[i])
        if cur:
            groups.append((cur_owner or "", cur))
        parts: list[str] = []
        for uid, claim_group in groups:
            head = ""
            unit = self._units.get(uid)
            if self._pool_header is not None and unit is not None:
                try:
                    head = str(self._pool_header(unit) or "")
                except Exception:  # noqa: BLE001 — a header hook must never break serving
                    logger.debug("pool_header raised — ignored", exc_info=True)
            parts.append((head + "\n" if head else "") + "\n".join("- " + c for c in claim_group))
        return "\n\n".join(parts)

    def _emit(self, kind: str, **fields: Any) -> None:
        """Deliver one structured freshness event to the ``on_event`` hook (v0.5). The hook
        must NEVER break serving — any exception it raises is swallowed (debug-logged)."""
        if self._on_event is None:
            return
        try:
            self._on_event({"event": kind, "ts": self._clock(), **fields})
        except Exception:  # noqa: BLE001 — observability must never take down a read
            logger.debug("on_event hook raised for %r — ignored", kind, exc_info=True)

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
        idx = self._fast_index()
        if idx is not None:
            _, _, _, _, _, _, _, c_meta, _ = idx
            _, _, claim_sims = self._fast_scores(idx, qe)
            for i, (text, uid, _atomic, fresh, row_ns) in enumerate(c_meta):
                if row_ns != ns or not fresh or not text:
                    continue
                scored.append(RecalledClaim(claim=text, score=float(claim_sims[i]), unit_id=uid))
        else:
            for unit in self._units.values():
                if unit.namespace != ns or not unit.is_fresh:
                    continue
                for text, emb in zip(self._claim_texts(unit.understanding),
                                     unit.claim_embeddings):
                    if not emb or not text:
                        continue
                    scored.append(
                        RecalledClaim(claim=text, score=cosine(qe, emb), unit_id=unit.id))
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

    def _bridge_claims(
        self, matched: Cognition, recalled: list[RecalledClaim], ns: str
    ) -> list[RecalledClaim]:
        """Bridge restart (the bench_multihop "(d)" arm, promoted into the library in v0.5).

        Rank OTHER fresh units by MaxSim of their claims to the MATCHED unit's own claim
        embeddings — expanding from the bridge entity the matched unit names, which is where
        hop-2 lives when it does not resemble the question. Returns the single best claim of
        each of the top ``bridge_limit`` units that are not already contributing to
        ``recalled`` (bridge = expansion to NEW units). Pure cosine over cached embeddings,
        zero LLM calls. NOTE: ``score`` here is bridge-similarity (cos to the matched unit's
        claims), NOT query-similarity — callers must never feed it into coverage."""
        seeds = [emb for emb in matched.claim_embeddings if emb]
        if not seeds:
            return []
        contributing = {rc.unit_id for rc in recalled} | {matched.id}
        seen_texts = {rc.claim for rc in recalled}
        candidates: list[tuple[float, RecalledClaim]] = []
        idx = self._fast_index()
        if idx is not None:
            _, uids, _, _, spans, _, CE, c_meta, dim = idx
            S = _np.asarray([s for s in seeds if len(s) == dim], dtype=_np.float64)
            if S.shape[0] and CE.shape[0]:
                Sn = _np.linalg.norm(S, axis=1, keepdims=True)
                S = S / _np.where(Sn > 0, Sn, 1.0)
                bridge_sims = (CE @ S.T).max(axis=1)
                for i, (uid, unit) in enumerate(self._units.items()):
                    if unit.namespace != ns or not unit.is_fresh or uid in contributing:
                        continue
                    start, end = spans[i]
                    best_score, best_text = 0.0, ""
                    for j in range(start, end):
                        text, _uid, atomic, _fresh, _ns2 = c_meta[j]
                        if not text or not atomic or text in seen_texts:
                            continue
                        score = float(bridge_sims[j])
                        if score > best_score:
                            best_score, best_text = score, text
                    if best_text:
                        candidates.append(
                            (best_score,
                             RecalledClaim(claim=best_text, score=best_score, unit_id=uid)))
            candidates.sort(key=lambda pair: pair[0], reverse=True)
            return [claim for _, claim in candidates[: self._bridge_limit]]
        for unit in self._units.values():
            if unit.namespace != ns or not unit.is_fresh or unit.id in contributing:
                continue
            # The embedding pool is [claims..., residuals..., summary]; the bridge serves
            # ATOMIC facts only — the summary would re-inflate tokens and blur provenance.
            atomic = {t for t in unit.understanding.get("claims") or [] if isinstance(t, str)}
            best_score, best_text = 0.0, ""
            for text, emb in zip(self._claim_texts(unit.understanding), unit.claim_embeddings):
                if not emb or not text or text not in atomic or text in seen_texts:
                    continue
                score = max(cosine(seed, emb) for seed in seeds)
                if score > best_score:
                    best_score, best_text = score, text
            if best_text:
                candidates.append(
                    (best_score, RecalledClaim(claim=best_text, score=best_score, unit_id=unit.id))
                )
        candidates.sort(key=lambda pair: pair[0], reverse=True)
        return [claim for _, claim in candidates[: self._bridge_limit]]

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
            self._emit("source_changed", artifact_id=event.artifact_id,
                       change_kind=event.kind, matched_units=0, dirtied=[], deleted=[])
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
        self._invalidated_units += len(result.dirtied) + len(result.deleted)
        self._emit(
            "source_changed",
            artifact_id=event.artifact_id,
            change_kind=event.kind,
            matched_units=result.matched_units,
            dirtied=list(result.dirtied),
            deleted=list(result.deleted),
            skipped_unchanged=len(result.skipped_unchanged),
        )
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
        now = self._clock()
        fresh_ages = [max(now - u.freshness_epoch, 0.0) for u in self._units.values() if u.is_fresh]
        return {
            "units": len(self._units),
            "tracked_artifacts": len(self._artifact_index),
            "reads": self._reads_total,
            "hits": hits,
            "escalations": self._reads_escalated,
            "escalation_rate": round(self._reads_escalated / hits, 3) if hits else 0.0,
            "hit_rate": round(hits / self._reads_total, 3) if self._reads_total else 0.0,
            # v0.5 — freshness observability: how often the cache REFUSED to serve stale
            # knowledge, how much got invalidated, and how old served knowledge actually is.
            "staleness_prevented": self._stale_reads_prevented,
            "invalidated_units": self._invalidated_units,
            "avg_age_at_serve_s": round(self._age_serve_sum / self._age_serve_n, 3)
            if self._age_serve_n else 0.0,
            "max_age_at_serve_s": round(self._age_serve_max, 3),
            "oldest_fresh_unit_s": round(max(fresh_ages), 3) if fresh_ages else 0.0,
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
