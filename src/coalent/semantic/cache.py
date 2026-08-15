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
import math
import re
import time
import warnings
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
from .pool import ClaimIndex, ClaimRef, LocalClaimIndex
from .ports import Chunk, Retriever, Synthesizer, Usage
from .store import CognitionStore
from .unit import Cognition, QueryKey, ResidualSpan

try:  # optional acceleration for pool serving at scale (``pip install coalent[fast]``).
    # The core stays zero-dependency: every numpy path has a pure-Python twin.
    import numpy as _np
except ImportError:  # pragma: no cover - environment-dependent
    _np = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_NUM = re.compile(r"\d")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
# v0.6 tier-2 residual spans — the build-time sentence-coverage audit's filters. A sentence is
# FACT-BEARING when it carries a number or a multi-word capitalized name; boilerplate (privacy/
# cookie/subscribe chrome, liveblog clock-stamp lines) is excluded even when number-bearing.
_SPAN_NAME = re.compile(r"\b[A-Z][\w&.'-]*(?:\s+[A-Z][\w&.'-]*)+")
_SPAN_BOILERPLATE = re.compile(
    r"(?i)privacy (?:policy|notice)|cookies?\b|subscri|newsletter|sign (?:up|in)\b|"
    r"log ?in\b|all rights reserved|terms of (?:service|use)"
    r"|^\s*\d{1,2}[:.]\d{2}\s*(?:a\.?m\.?|p\.?m\.?|[A-Z]{2,4}\b)"
)
_SPAN_MIN_WORDS = 6            # shorter fragments are headers/captions, not fact sentences
_SPAN_DEDUP_SIM = 0.95         # within-unit near-dup bar for the append-only repair merge
_SPANS_PER_READ = 2            # side-channel cap: at most 2 residual spans served per read
_SPAN_CAP = 12                 # stored spans per unit, most-clearly-uncovered kept first
_SPAN_LABEL = "[source excerpt]"
_READ_LOG_CAP = 64             # ring buffer of recent reads report_refusal() can look up
_MAX_KEYS_PER_UNIT = 8         # query-key cap per unit (evicts the lowest-hits existing key)
# The HARD facts a candidate span must add over the claims (eyeball-audit fix: ~50% of raw
# captures were 0.55-0.61 phrasing gaps — the sentence was covered by a REPHRASED claim).
_HARD_NUM = re.compile(r"\b\d[\d,.]*%?\b")
# A stranded <=3-char token before a blank line ("ET\n\nPotential Super Bowl...") — the
# sentence splitter's orphan; stripped before filtering.
_SPAN_ORPHAN = re.compile(r"^\S{1,3}\s*\n\s*\n\s*")

# v0.6 pool read path — module CONSTANTS, deliberately not knobs (spec §2.2): the pool
# path's whole point is deleting tunables, so operating points live here, test-pinned.
_POOL_STAGE1_RAW = 600            # cosine candidates scanned before dedup
_POOL_STAGE1_WIDTH = 400          # unique candidates after dedup = the rerank window
_POOL_DEDUP_SIM = 0.95            # near-dup cosine threshold (numpy path only)
_MAX_BUILDS_PER_READ = 3          # synthesis ops (rebuilds + new builds) per read
_MAX_TTL_REVALIDATIONS_PER_READ = 3   # revalidator-backed TTL refreshes per read
_POOL_GATE_MARGIN = 0.02
_POOL_GATE_CEILING_MARGIN = 0.27  # adaptive gate ceiling = cov_default + 0.27
_PUREPY_POOL_WARN_ROWS = 2000     # pure-python pool scan warning threshold


def _est_tokens(s: str) -> int:
    """The serving token estimate (~4 chars/token), never zero — the ONE estimator the
    packing contract, the TTL candidate head, and the RAG-floor cap all share."""
    return max(1, len(s) // 4)


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
class _PoolHit:
    """One pool-serving candidate row (v0.6 internal): the pre-rerank query cosine —
    which every gate decision reads — plus the exact owner row it came from."""

    score: float          # cosine(query, claim) — ALWAYS the pre-rerank score
    unit_id: str
    claim_idx: int        # position in the owner's claim list at add time
    text: str


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
    pool: list[RecalledClaim] = field(default_factory=list)  # v0.6 pool path: served claims in
    #                             served order; score is ALWAYS the query cosine (never a rerank
    #                             score); unit_id is the owning unit. Empty on the unit path.
    usage: Usage | None = None  # synth tokens for THIS read; None when no LLM ran (a hit, OR a
    #                             usage-less provider) — use `cache_hit` as the authoritative hit signal
    needs_retrieval: bool = False  # S3 hint: cache under-covered even after recall — you MAY retry/widen
    read_id: str = ""           # v0.6: this read's id — hand it to report_refusal() when YOUR
    #                             answerer refuses over the served payload (the behavioral
    #                             residual-span fallback; requires residual_spans=True)

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
        serve_budget: int | None = None,
        pool_header: Callable[[Cognition], str] | None = None,
        read_path: str = "unit",
        serve_gate: float | None = None,
        reranker: Callable[[str, list[str]], list[float]] | None = None,
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
        residual_spans: bool = False,
        span_tau: float = 0.62,
        span_margin: float = 0.0,
        span_serve_floor: float = 0.35,
        lossy_threshold: int = 2,
        query_keys: bool = False,
        key_floor: float = 0.85,
        strategy: str = ContextStrategy.CONTEXT_FIRST,
        store: CognitionStore | None = None,
        freshness: FreshnessPolicy | None = None,
        clock: Callable[[], float] = time.time,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        claim_index: "ClaimIndex | Callable[[str], ClaimIndex] | None" = None,
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
        # v0.5 — adaptive hit gate (warm-up finding: match scores INFLATE as units
        # accumulate, so a fixed threshold sinks below the noise floor and the cache
        # 'absorbs' everything, including unanswerable queries). Self-calibrates against
        # the cache's own cross-unit score distribution; opt-in.
        self._adaptive_hit = adaptive_hit
        self._noise_ceiling = 0.0
        self._builds_since_calib = 0
        # v0.5 gate v2 — the REUSE channel: a query whose SEED similarity to a cached unit is
        # near-identity is the same question asked again; it must hit regardless of the noise
        # bar (replay finding: the adaptive bar killed revisits; the blend diluted the seed
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
        # v0.6 — the pool-first read path (spec §2). Opt-in via read_path="pool"; the default
        # "unit" is byte-identical v0.5. Pool mode REQUIRES a semantic embedder (constructor-time
        # guard): under the lexical HashingEmbedder, claim cosine collapses to keyword overlap and
        # the whole gate/ranking substrate would amplify keyword coincidence.
        if read_path not in ("unit", "pool"):
            raise ValueError(f"read_path must be 'unit' or 'pool', got {read_path!r}")
        if read_path == "pool" and isinstance(self._embedder, HashingEmbedder):
            raise ValueError(
                "read_path='pool' requires a semantic embedder — set OPENAI_API_KEY or pass "
                "embedder=...; claim-cosine collapses to keyword overlap under HashingEmbedder"
            )
        if query_keys and read_path != "pool":
            # Fail LOUD at construction: the key overlay serves only through the pool read
            # path — on the unit path keys could attach + confirm yet structurally never
            # fire (the exact silent-failure class a measured forensic run paid for).
            raise ValueError("query_keys requires read_path='pool'")
        if read_path == "pool" and pool_header is None:
            # Measured ladder (n=605 graded): bare default 0.68 vs metadata callable 0.73 —
            # the gap is source-identity questions honestly refusing. Warn, don't fail:
            # the default header (unit title text) is legitimate when sources are homogeneous.
            warnings.warn(
                "read_path='pool' without pool_header: payload attribution falls back to unit "
                "title text; per-source questions may refuse. Pass pool_header returning "
                "'[title | source | date]' for full attribution (see the v0.6 upgrade guide).",
                stacklevel=2,
            )
        self._read_path = read_path
        # serve_gate: explicit float = ABSOLUTE (adaptation disabled entirely — the operator's
        # off-ramp for exact, reproducible benches); None = adaptive against the pool's own
        # null-shaped noise ceiling (§2.3), clamped at cov_default + 0.27.
        self._serve_gate = serve_gate
        # Rerank hook (BYO): influences SERVING ORDER only — the serve/build/floor decisions and
        # pool_coverage always read the pre-rerank cosine, so a bad reranker can degrade order
        # but never cause a false serve, a skipped build, or a broken null refusal.
        self._reranker = reranker
        # serve_budget default split: None -> 600 on the unit path (v0.5 preserved verbatim) and
        # 1000 on the pool path (reproduces the measured ~1036-token operating point under the
        # strict header-counting packing contract). Explicit values always win.
        if serve_budget is not None and serve_budget <= 0:
            raise ValueError(f"serve_budget must be positive, got {serve_budget}")
        self._serve_budget = (serve_budget if serve_budget is not None
                              else (1000 if read_path == "pool" else 600))
        self._pool_header = pool_header
        self._pool_state: tuple[Any, ...] | None = None   # lazy (marker, texts, unit_ids, embs)
        # v0.6 — pool read-path state: adaptive-gate noise ceiling, lazy per-namespace index
        # hydration, and the observability counters (§4.5) that make gate miscalibration a
        # dashboard number instead of a silent duplicate-build tax.
        self._pool_noise_ceiling = 0.0
        self._pool_hydrated: set[str] = set()
        self._pool_scan_slow = False
        self._pool_serves = 0
        self._probe_reads = 0
        self._admission_reuses = 0
        self._retrievals = 0
        self._rebuilds_by_read = 0
        self._builds_by_gap = 0
        self._pool_units_sum = 0
        self._pool_payloads = 0
        self._last_collapsed: dict[tuple[str, int], tuple[str, int]] = {}
        # v0.6 — invalidation counters replacing the (len, xor-hash) pool marker, whose
        # in-place-rebuild cancellation (dirty -> rebuild -> fresh, same id & count restores the
        # prior marker) could serve OLD claim texts from a memoized pool. Two monotone counters,
        # one choke point each: ``_rows_epoch`` bumps ONLY when rows change (build / evict /
        # backfill) and NEVER cancels; ``_status_gen`` bumps on any freshness flip. Cache-side
        # lazy states (pool serving, the SE/UE/CE fast scan index) key on BOTH, so neither a
        # same-size rebuild nor a stale flip can be masked by a colliding marker.
        self._rows_epoch = 0
        self._status_gen = 0
        # v0.6 — the claim-pool storage port (one index per namespace). None uses the built-in
        # LocalClaimIndex; a bare instance is single-namespace ONLY (a second namespace raises,
        # telling the caller to pass a factory) so it can never leak claims across namespaces; a
        # ``Callable[[str], ClaimIndex]`` factory mints one index per namespace.
        self._claim_index_arg = claim_index
        self._claim_indexes: dict[str, ClaimIndex] = {}
        self._bare_index_ns: str | None = None
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
        # v0.6 (opt-in, OFF — nothing below runs when False; the default path is byte-identical):
        # TIER-2 RESIDUAL SPANS, the measured extraction-loss net. At build, a sentence-coverage
        # audit retains fact-bearing source sentences the extractor missed (max claim cosine <
        # span_tau) as spans ON THE UNIT — never as pool rows (news pools stay lean). At read, a
        # span that outranks every fresh claim by span_margin serves as a labeled side channel;
        # span serves + raw-fallback escalations count toward lossy_threshold, and a lossy-marked
        # unit repairs APPEND-ONLY on its next rebuild touch (claims are never lost while the
        # source hash is unchanged — coverage is monotone).
        self._residual_spans = residual_spans
        self._span_tau = span_tau
        self._span_margin = span_margin
        self._span_serve_floor = span_serve_floor
        self._lossy_threshold = lossy_threshold
        self._span_state: tuple[Any, ...] | None = None   # lazy per-ns span matrix (epoch-keyed)
        # The behavioral fallback channel: every read gets a deterministic read_id; the last
        # _READ_LOG_CAP reads are remembered (query embedding, ns, served payload facts) so
        # report_refusal(read_id) can answer a refusal with a residual-span retry payload.
        self._read_seq = 0
        self._read_log: dict[
            str, tuple[tuple[float, ...], str, tuple[str, ...], tuple[str, ...], int, int]
        ] = {}
        # v0.6 (opt-in, OFF — its own switch on top of residual_spans): QUERY KEYS, the
        # behavioral alternate retrieval keys (harness-measured: affected-question gold rank
        # median 139 -> 0, top-20 0% -> 99% on paraphrases, controls unharmed). A refusal
        # attaches the read's query embedding as a PROVISIONAL key on the span's owner;
        # report_success() confirms it durable; the repair promotion transfers it to the
        # claim row. At read time, keys overlay the pool scan: a claim's serving score is
        # max(content_sim, key_sim) where a key counts ONLY at/above key_floor — below the
        # floor the key row is ignored entirely (the v0.2-regression guard in the rule).
        self._query_keys = query_keys
        self._key_floor = key_floor
        self._key_gen = 0                                 # bumps on attach/expire/evict
        self._key_state: tuple[Any, ...] | None = None    # lazy per-ns key matrix
        self._keys_by_read: dict[str, list[str]] = {}     # read_id -> units holding its keys
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

    def _dirty(self, unit: Cognition) -> None:
        """Mark a unit stale through the ONE freshness choke point: flip status and bump
        ``_status_gen`` so every cache-side lazy state (pool serving, the fast scan index)
        that snapshots freshness is invalidated on the next read. Callers persist as before.

        Pool mode also masks the unit's claim rows in its namespace index. The in-proc state
        is made consistent FIRST (status + generation), so a persistent adapter's ``mask``
        failure — which MUST be loud (a shared index serving stale to other processes) —
        propagates only after this process can no longer serve the stale rows."""
        unit.mark_dirty()
        self._status_gen += 1
        if self._read_path == "pool":
            idx = self._existing_index(unit.namespace)
            if idx is not None:
                idx.mask(unit.id)

    def _mark_fresh(self, unit: Cognition) -> None:
        unit.mark_fresh()
        unit.freshness_epoch = self._clock()
        self._status_gen += 1
        if self._read_path == "pool":
            idx = self._existing_index(unit.namespace)
            if idx is not None:
                idx.unmask(unit.id)

    def _existing_index(self, ns: str) -> ClaimIndex | None:
        """The claim index already minted for ``ns`` — or None. Used by freshness flips and
        eviction, which must never MINT an index (and never trip the bare-instance namespace
        guard) just to mask rows that were never added."""
        arg = self._claim_index_arg
        if isinstance(arg, ClaimIndex):
            return arg if self._bare_index_ns == ns else None
        return self._claim_indexes.get(ns)

    def _unit_is_fresh(self, unit_id: str) -> bool:
        """Live freshness authority for a claim index's pull-mask: a unit is servable iff it
        exists and is FRESH — evaluated per scan, never snapshotted into an index row."""
        unit = self._units.get(unit_id)
        return unit is not None and unit.is_fresh

    def _resolve_claim_index(self, ns: str) -> ClaimIndex:
        """Resolve the :class:`ClaimIndex` for a namespace — the single choke point both the
        build (add) and read (search) paths funnel through, so the namespace guard applies to
        both. A factory (or the built-in default) mints one index per namespace; a bare instance
        is bound to the first namespace it serves and a second namespace raises, since sharing one
        instance across namespaces would leak claims (the D1 defect)."""
        arg = self._claim_index_arg
        if isinstance(arg, ClaimIndex):
            # Bare instance: single-namespace only (checked on BOTH add and search via this method).
            if self._bare_index_ns is None:
                self._bare_index_ns = ns
            elif self._bare_index_ns != ns:
                raise ValueError(
                    f"a bare ClaimIndex instance is single-namespace only (bound to "
                    f"{self._bare_index_ns!r}, now asked for {ns!r}) — pass a factory "
                    "claim_index=lambda ns: LocalClaimIndex(...) so each namespace gets its own"
                )
            return arg
        idx = self._claim_indexes.get(ns)
        if idx is None:
            if callable(arg):
                idx = arg(ns)
            else:
                idx = LocalClaimIndex(fresh_of=self._unit_is_fresh)
            self._claim_indexes[ns] = idx
        return idx

    @staticmethod
    def _sum_usage(a: Usage | None, b: Usage | None) -> Usage | None:
        """Combine two synthesis-call usages so a multi-build read reports the TOTAL it spent
        (the D4 fix). None is the additive identity; the first known model label is kept."""
        if a is None:
            return b
        if b is None:
            return a
        return Usage(
            prompt_tokens=a.prompt_tokens + b.prompt_tokens,
            completion_tokens=a.completion_tokens + b.completion_tokens,
            model=a.model or b.model,
            cost=a.cost + b.cost,
        )

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
        if self._read_path == "pool":
            # v0.6 — the pool-first read path (opt-in). The unit path below stays byte-identical.
            return self._pool_read(query, namespace=namespace, related=related, strategy=strategy)
        strat = strategy or self._strategy
        ns = namespace or ""
        qe = tuple(self._embedder.embed(query))
        self._read_seq += 1
        read_id = f"read-{self._read_seq}"   # deterministic (a monotonic counter string)
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
                            self._dirty(u_)
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
            # transient build failures were being cached and served as empty context.
            if unit.is_fresh and unit.understanding.get("_synthesis_failed"):
                self._dirty(unit)
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

        if self._residual_spans:
            self._log_read(read_id, qe, ns, (unit.id,), (), 0)
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
            read_id=read_id,
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
        a falling build rate stays a REAL signal (Grid-C finding)."""
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
        """Lazy q-independent structures for the vectorized scans; keyed on
        ``(_rows_epoch, _status_gen)`` — the fast index snapshots ``is_fresh`` into its claim
        metadata, so a freshness flip (``_status_gen``) MUST rebuild it as well as a row change
        (``_rows_epoch``). None when numpy is absent."""
        if not self._fast_enabled or _np is None:
            return None
        marker = (self._rows_epoch, self._status_gen)
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
        q: Any = _np.asarray(qe if len(qe) == dim else (0.0,) * dim, dtype=_np.float64)
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
        self._retrievals += 1     # the query-shaped retrieval counter (pool invariant: <= 1/read)
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
        # into one lossy understanding (the digest/multi-source finding).
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
        # D4 fix: a multi-source read runs SEVERAL synthesis calls; ``read_usage`` (-> Result.usage)
        # must report the TOTAL spent, not just the dominant build. Sum every _build_unit usage;
        # track whether the dominant unit itself was built with a dedicated flag (``usage`` is no
        # longer a proxy for that, now that sibling/rebuild calls also contribute to it).
        usage: Usage | None = None
        first = True
        dominant_built = False
        for artifact_id, chs, existing in todo:
            built_chunks = self._widen_group(artifact_id, chs)
            if existing is not None:                  # widen-rebuild in place: no duplicate
                usage = self._sum_usage(usage,
                                        self._build_unit(existing, query, qe, built_chunks))
                self._emit("unit_rebuilt", unit_id=existing.id, reason="widen",
                           artifact_id=artifact_id)
                if first:
                    first = False
                continue
            if first:
                usage = self._sum_usage(usage,
                                        self._build_unit(unit, query, qe, built_chunks))
                first = False
                dominant_built = True
                continue
            sibling = self._new_unit(f"{query} · {artifact_id}", qe, ns)
            usage = self._sum_usage(usage,
                                    self._build_unit(sibling, query, qe, built_chunks))
            self._units[sibling.id] = sibling
            self._emit("unit_built", unit_id=sibling.id, reason="split",
                       artifact_id=artifact_id)
        if not dominant_built:                        # dominant was a widen-rebuild
            usage = self._sum_usage(usage, self._build_unit(
                unit, query, qe, self._widen_group(todo[0][0], todo[0][1])))
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
        # v0.6 tier-2 residual spans (opt-in): the build-time sentence-coverage audit — retain
        # fact-bearing evidence sentences the extraction missed as spans ON the unit (provenance
        # captured now, never re-derived). A (re)build also clears the lossy signals: whatever
        # this unit failed to cover before, it was just re-extracted against current sources.
        if self._residual_spans:
            unit.residual_spans = (
                self._capture_residual_spans(
                    chunks, self._claim_texts(understanding), unit.claim_embeddings)
                if synthesis.ok else ()
            )
            unit.span_hits = 0
            unit.lossy = False
        if self._query_keys:
            self._rebind_keys(unit)   # claim positions regenerated — re-point every key
        unit.synth_tokens = synthesis.usage.total_tokens if synthesis.usage is not None else 0
        self._builds_since_calib += 1
        unit.touch()  # a build, not a hit -> no query recorded
        self._mark_fresh(unit)
        self._reindex(unit)
        self._persist(unit)
        self._rows_epoch += 1     # the unit's claim rows changed — monotone, never cancels
        self._index_unit_rows(unit)   # pool mode: incremental per-unit row replacement
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

    # ------------------------------------------ tier-2 residual spans (v0.6, opt-in)
    def _capture_residual_spans(
        self,
        chunks: list[Chunk],
        claim_texts: list[str],
        claim_embeddings: tuple[tuple[float, ...], ...],
    ) -> tuple[ResidualSpan, ...]:
        """The build-time sentence-coverage audit (the preventive detector for the measured
        50-84% harmful extraction loss). Splits the unit's retained evidence into sentences
        (>= _SPAN_MIN_WORDS words, starting with a capitalized word or digit, no embedded
        blank line; a stranded <=3-char splitter orphan is stripped first), keeps the
        FACT-BEARING ones (a number or a multi-word capitalized name; boilerplate excluded)
        that carry a HARD FACT absent from every claim text — a multi-char number no claim
        contains, or a capitalized name none mentions (the phrasing-gap false-alarm killer:
        a sentence whose facts all live in rephrased claims is covered, not lost). Survivors
        are embedded in ONE batch call; those under ``span_tau`` max claim cosine become
        tier-2 spans — verbatim text + artifact + evidence-chunk index, provenance captured
        at build and never re-derived — capped at ``_SPAN_CAP`` most-clearly-uncovered per
        unit. Spans are NEVER added to the claim pool (zero crowding); they serve only
        through the side channel."""
        claim_blob = " ".join(claim_texts).lower()
        cand: list[tuple[str, str, int]] = []
        seen: set[str] = set()
        for idx, chunk in enumerate(chunks):
            for raw in _SENT_SPLIT.split(chunk.text):
                s = raw.strip()
                m = _SPAN_ORPHAN.match(s)             # "ET\n\nPotential Super Bowl..." relic
                if m:
                    s = s[m.end():].strip()
                if not s or "\n\n" in s or s in seen:
                    continue
                if len(s.split()) < _SPAN_MIN_WORDS:
                    continue
                if not (s[0].isupper() or s[0].isdigit()):
                    continue                          # segmentation orphans never start caps
                if not (_NUM.search(s) or _SPAN_NAME.search(s)):
                    continue                          # fact-bearing sentences only
                if _SPAN_BOILERPLATE.search(s):
                    continue                          # privacy/cookie/subscribe/liveblog chrome
                if not self._has_absent_hard_fact(s, claim_blob):
                    continue                          # every hard fact already lives in a claim
                seen.add(s)
                cand.append((s, chunk.artifact_id, idx))
        if not cand:
            return ()
        scored: list[tuple[float, ResidualSpan]] = []
        for (text, artifact_id, idx), raw_emb in zip(
            cand, embed_texts(self._embedder, [c[0] for c in cand])
        ):
            emb = tuple(raw_emb)
            if not emb or not any(emb):
                continue
            covered = max((cosine(emb, ce) for ce in claim_embeddings if ce), default=0.0)
            if covered < self._span_tau:
                scored.append((covered, ResidualSpan(text=text, artifact_id=artifact_id,
                                                     chunk_idx=idx, embedding=emb)))
        scored.sort(key=lambda cs: cs[0])   # most clearly uncovered first (stable: doc order)
        return tuple(span for _covered, span in scored[:_SPAN_CAP])

    @staticmethod
    def _has_absent_hard_fact(sentence: str, claim_blob: str) -> bool:
        """True when the sentence carries at least one HARD fact no claim text contains: a
        multi-char number (substring check against the lowercased concatenated claims) or a
        multi-word capitalized name (case-insensitive containment). A sentence whose facts
        all appear in the claims is a phrasing gap, not extraction loss — the eyeball-audit
        false-alarm mode this predicate kills."""
        for num in _HARD_NUM.findall(sentence):
            if len(num) > 1 and num not in claim_blob:
                return True
        for name in _SPAN_NAME.findall(sentence):
            if name.lower() not in claim_blob:
                return True
        return False

    def _bump_lossy(self, unit: Cognition, reason: str) -> None:
        """One lossy signal against a unit (a residual-span serve, or a raw-fallback
        escalation over its source). Crossing ``lossy_threshold`` marks the unit lossy —
        ONCE — so the next rebuild touch repairs it via the append-only path."""
        unit.span_hits += 1
        if not unit.lossy and unit.span_hits >= self._lossy_threshold:
            unit.lossy = True
            self._emit("unit_marked_lossy", unit_id=unit.id, reason=reason)
        self._persist(unit)

    def _repair_unit(
        self, unit: Cognition, query: str, qe: tuple[float, ...], chunks: list[Chunk]
    ) -> Usage | None:
        """Repair a lossy-marked unit via the existing rebuild-in-place machinery, with THE
        APPEND-ONLY INVARIANT: when the source content is provably UNCHANGED since build
        (every previously-cited span's content hash for a re-fetched artifact is still present
        in the new chunks), the merged claim set is union(old, new) deduped within-unit at
        ``_SPAN_DEDUP_SIM`` — old claims are NEVER lost, coverage is monotone. A CHANGED hash
        falls through to the full replace the rebuild already performed (stale behavior).
        Emits ``unit_repaired{unit_id, added_claims, kept_claims}`` either way."""
        old_texts_all, old_embs_all = self._atomic_rows(unit)
        old_atomic = [(t, e) for t, e in zip(old_texts_all, old_embs_all) if t and any(e)]
        new_hashes: dict[str, set[str]] = {}
        for c in chunks:
            h = c.content_hash or SourceSpan.from_text(c.artifact_id, c.text).content_hash
            new_hashes.setdefault(c.artifact_id, set()).add(h)
        prov_spans = unit.provenance.source_spans
        unchanged = (
            any(span.artifact_id in new_hashes for span in prov_spans)   # can't prove -> changed
            and all(
                span.content_hash in new_hashes[span.artifact_id]
                for span in prov_spans
                if span.artifact_id in new_hashes
            )
        )
        usage = self._build_unit(unit, query, qe, chunks)   # the rebuild-in-place machinery
        if not unchanged or not old_atomic:
            new_texts, new_embs = self._atomic_rows(unit)
            n_new = sum(1 for t, e in zip(new_texts, new_embs) if t and any(e))
            self._emit("unit_repaired", unit_id=unit.id,
                       added_claims=n_new, kept_claims=0)   # full replace (stale behavior)
            return usage
        # Append-only merge: old claims first (never lost), then new claims that are not
        # near-duplicates of anything kept. Then re-key the unit and re-index its pool rows
        # (add() has REPLACE semantics, so the pool holds no duplicate rows after repair).
        kept_texts = [t for t, _e in old_atomic]
        kept_embs = [e for _t, e in old_atomic]
        kept_set = set(kept_texts)
        added = 0
        new_texts, new_embs = self._atomic_rows(unit)
        for t, e in zip(new_texts, new_embs):
            if not t or not any(e) or t in kept_set:
                continue
            if max((cosine(e, ke) for ke in kept_embs), default=0.0) >= _SPAN_DEDUP_SIM:
                continue
            kept_texts.append(t)
            kept_embs.append(e)
            kept_set.add(t)
            added += 1
        if self._query_keys and unit.query_keys:
            # REAL span->claim promotion (the binding heal): a keyed span's VERBATIM source
            # sentence joins the merged claim set — append-only consistent (it is retained
            # source text), within-unit 0.95 dedup applies — so _rebind_keys below binds the
            # key to a real claim row by guaranteed-verbatim identity and the span-serving
            # row self-retires for it. Real LLM extraction never reproduces the sentence
            # byte-for-byte; promotion must not depend on it.
            promo = [t for t in dict.fromkeys(k.span_text for k in unit.query_keys)
                     if t and t not in kept_set]
            for t, raw_e in zip(promo, embed_texts(self._embedder, promo) if promo else []):
                e = tuple(raw_e)
                if not e or not any(e):
                    continue
                if max((cosine(e, ke) for ke in kept_embs), default=0.0) >= _SPAN_DEDUP_SIM:
                    continue
                kept_texts.append(t)
                kept_embs.append(e)
                kept_set.add(t)
                added += 1
        understanding = dict(unit.understanding)
        understanding["claims"] = kept_texts
        unit.understanding = understanding
        unit.understanding_embedding, unit.claim_embeddings = self._cognition_embeddings(
            understanding
        )
        # Re-audit the spans against the MERGED claim set (a merged-in claim may now cover
        # what the fresh extraction alone missed).
        unit.residual_spans = self._capture_residual_spans(
            list(unit.evidence), self._claim_texts(understanding), unit.claim_embeddings)
        if self._query_keys:
            self._rebind_keys(unit)   # the merge moved claim positions — re-point the keys
        self._persist(unit)
        self._rows_epoch += 1     # the merge changed the unit's rows again — monotone
        self._index_unit_rows(unit)
        self._emit("unit_repaired", unit_id=unit.id,
                   added_claims=added, kept_claims=len(old_atomic))
        return usage

    def _backfill_cognition(self, unit: Cognition) -> None:
        """One-time upgrade of a pre-v0.3 unit: compute + persist its embeddings."""
        unit.understanding_embedding, unit.claim_embeddings = self._cognition_embeddings(
            unit.understanding
        )
        self._persist(unit)
        self._rows_epoch += 1     # the unit's claim rows materialized — monotone
        self._index_unit_rows(unit)   # pool mode: incremental per-unit row replacement

    # ------------------------------------------------ context intelligence
    def _semantic_coverage(self, qe: tuple[float, ...], unit: Cognition) -> float:
        """How well the unit's best single claim addresses THIS query: max cosine of the
        query against each per-claim embedding. With no claim embeddings the verdict
        depends on whether there is anything SERVABLE at all: a structured/passthrough
        unit with real content keeps the benign 1.0 ("can't prove a gap — don't
        penalize"), but an EMPTY unit (failed synthesis: no claims, no summary) reports
        0.0 — v0.5 fix for the -discovered poison where hollow units claimed perfect
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
        # Keyed on ``(ns, _rows_epoch, _status_gen)``: the cached pool snapshots BOTH the
        # servable rows (``_rows_epoch``) and their freshness (``_status_gen``). Replacing the
        # old (len, xor-hash) marker closes the in-place-rebuild cancellation hole — a monotone
        # ``_rows_epoch`` cannot be restored to a prior value by a same-size rebuild.
        marker = (ns, self._rows_epoch, self._status_gen)
        state = self._pool_state
        if state is None or state[0] != marker:
            texts, owners, embs = self._pool_rows(ns)
            matrix: Any = None
            if _np is not None and embs:
                matrix = _np.asarray(embs, dtype=_np.float64)
                matrix /= _np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
            state = (marker, texts, owners, embs, matrix)
            self._pool_state = state
        _, texts, owners, embs, matrix = state
        if not texts:
            return ""
        if matrix is not None:
            q: Any = _np.asarray(qe, dtype=_np.float64)
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

    # ------------------------------------------------ pool-first read path (v0.6, read_path="pool")
    def _pool_read(
        self,
        query: str,
        *,
        namespace: str | None,
        related: int,
        strategy: str | None,
    ) -> Result:
        """The claim-pool-first read (spec §2.4, P0–P8): every read is answered by
        budget-packing the global fresh-claim pool; units remain the ownership /
        freshness / build / provenance skeleton. At most ONE query-shaped retrieval
        per read; gate/floor decisions read the PRE-rerank cosine only."""
        # P0 — VALIDATE / EMBED
        strat = strategy or self._strategy
        ns = namespace or ""
        qe = tuple(self._embedder.embed(query))
        self._read_seq += 1
        read_id = f"read-{self._read_seq}"   # deterministic (a monotonic counter string)
        self._reads_total += 1
        read_usage: Usage | None = None
        synthesis_ran = False
        retrieval_ran = False
        probe_chunks: list[Chunk] = []
        reuse_evidence: list[Chunk] = []
        force_needs_retrieval = False
        gate = self._effective_pool_gate()

        # P1 — REUSE: near-identity SEED similarity = the same question asked again; it must
        # serve regardless of the noise bar. A STALE reuse match rebuilds in place FIRST
        # (consuming the read's single query-shaped retrieval), so the serve is never stale.
        forced_serve = False
        reuse_unit: Cognition | None = None
        best_reuse = 0.0
        for u in self._units.values():
            if u.namespace != ns or not u.query_embedding:
                continue
            s = cosine(qe, u.query_embedding)
            if s > best_reuse:
                best_reuse, reuse_unit = s, u
        if reuse_unit is not None and best_reuse >= self._reuse_threshold:
            forced_serve = True
            self._refresh_if_expired(reuse_unit)
            if not reuse_unit.is_fresh:
                self._stale_reads_prevented += 1
                self._emit(
                    "stale_read_prevented",
                    unit_id=reuse_unit.id,
                    query=query,
                    unit_age_s=round(max(self._clock() - reuse_unit.freshness_epoch, 0.0), 3),
                )
                read_usage = self._sum_usage(
                    read_usage, self._materialize_into(reuse_unit, query, qe, ns))
                synthesis_ran = True
                retrieval_ran = True
                reuse_evidence = list(reuse_unit.evidence)
                self._emit(
                    "rebuild_triggered_by_read",
                    unit_id=reuse_unit.id,
                    artifact_id=next(iter(sorted(reuse_unit.provenance.artifact_ids())), ""),
                    reason="reuse_stale",
                )
                self._rebuilds_by_read += 1
            elif self._residual_spans and reuse_unit.lossy:
                # Lossy repair on touch (tier-2 loop): the near-identity revisit IS this
                # unit's next touch — repair it FIRST via the append-only rebuild (consuming
                # the read's single retrieval), so the serve is post-repair. Only the unit's
                # OWN sources feed the repair; a probe with none of them skips (net stays up).
                probe_chunks = self._retrieve(query, ns)
                retrieval_ran = True
                own = reuse_unit.provenance.artifact_ids()
                own_chunks = [c for c in probe_chunks if c.artifact_id in own]
                if own_chunks:
                    read_usage = self._sum_usage(
                        read_usage, self._repair_unit(reuse_unit, query, qe, own_chunks))
                    synthesis_ran = True
                    reuse_evidence = list(reuse_unit.evidence)
                    self._rebuilds_by_read += 1
            reuse_unit.touch(
                query if self._learn_behavior else None, max_queries=self._max_hit_queries)

        # P2 — POOL SCAN via the ClaimIndex; freshness is a live pull-mask, never cached in rows.
        index = self._pool_index(ns)
        if _np is None and len(index) > _PUREPY_POOL_WARN_ROWS and not self._pool_scan_slow:
            self._pool_scan_slow = True
            logger.warning(
                "pure-python pool scan over %d rows — install coalent[fast] for numpy",
                len(index))
        cands = self._index_search(index, qe, _POOL_STAGE1_RAW)
        fresh_rows = [(s, ref) for s, ref, fresh in cands if fresh]
        stale_hits = [(s, ref) for s, ref, fresh in cands if not fresh and s >= gate]
        if stale_hits:  # telemetry only — include_stale MAY be unsupported (event won't fire)
            self._emit(
                "pool_masked_stale",
                n_rows=len(stale_hits),
                units=sorted({ref.unit_id for _, ref in stale_hits})[:8],
                top_stale_score=round(max(s for s, _ in stale_hits), 4),
            )

        # P3 — bounded TTL: clock-compare the candidate-head owners only; at most
        # _MAX_TTL_REVALIDATIONS_PER_READ revalidator calls; at most ONE re-scan.
        ttl_masked: set[str] = set()
        if (self._freshness is not None and self._freshness.max_age is not None
                and fresh_rows):
            ttl_masked, flipped = self._pool_ttl(fresh_rows)
            if flipped:
                cands = self._index_search(index, qe, _POOL_STAGE1_RAW)
                fresh_rows = [(s, ref) for s, ref, fresh in cands if fresh]
            if ttl_masked:
                fresh_rows = [(s, ref) for s, ref in fresh_rows
                              if ref.unit_id not in ttl_masked]
        key_fired: set[int] = set()
        fresh_rows = self._apply_query_keys(ns, qe, fresh_rows, ttl_masked, key_fired)
        cov0 = fresh_rows[0][0] if fresh_rows else 0.0

        # P4 — GATE (pre-rerank cosine only; a reranker can never influence this decision).
        # The gate arbitrates among candidates; ZERO fresh candidates is never a serve.
        # Without the fresh_rows guard, an explicit serve_gate=0.0 on a cold/empty pool
        # passes 0.0 >= 0.0 and serves empty forever, never triggering the gap build.
        outcome = "reuse" if forced_serve else (
            "serve" if fresh_rows and cov0 >= gate else "build")
        self._emit("pool_gate", coverage=round(cov0, 4), gate=round(gate, 4), outcome=outcome)
        if outcome != "build":
            self._pool_serves += 1

        # P5 — BUILD: one probe retrieval, classified per artifact (CONTAINED / STALE-OWNED /
        # THIN / NOVEL), capped at _MAX_BUILDS_PER_READ synthesis ops. Admission is STRUCTURAL:
        # all-contained probes serve with zero synthesis (provenance PROVES coverage).
        if outcome == "build":
            probe_chunks = self._retrieve(query, ns)   # THE single query-shaped retrieval
            retrieval_ran = True
            if not probe_chunks:
                force_needs_retrieval = True           # cold start: serve what the pool has
            else:
                built_usage, ran, touched = self._pool_probe_build(
                    query, qe, ns, probe_chunks, cov0, gate)
                read_usage = self._sum_usage(read_usage, built_usage)
                if ran:
                    synthesis_ran = True
                    ttl_masked -= touched              # a rebuilt owner is verifiably fresh
                    cands = self._index_search(index, qe, _POOL_STAGE1_RAW)
                    fresh_rows = [(s, ref) for s, ref, fresh in cands
                                  if fresh and ref.unit_id not in ttl_masked]
        fresh_rows = self._apply_query_keys(ns, qe, fresh_rows, ttl_masked, key_fired)
        final_cov = fresh_rows[0][0] if fresh_rows else 0.0

        # Serving pipeline (order only — decisions above already used pre-rerank cosine).
        kept = self._pool_candidates(fresh_rows)
        ordered, reranked, rerank_ms = self._pool_rank(query, kept)
        picked, est_used, headers = self._pool_pack(ordered)
        served_texts = [h.text for h in picked]

        # Tier-2 side channel (opt-in residual_spans): under the ranking-anomaly rule, a
        # fresh unit's residual span that outranks EVERY fresh claim serves as a labeled
        # excerpt inside the same budget. In-memory scoring only — never a retrieval; the
        # comparison reads the pre-S2 claim cosine (final_cov here) so both sides are cosines.
        served_spans: list[tuple[str, str]] = []
        if self._residual_spans:
            served_spans, est_used, headers = self._serve_residual_spans(
                ns, qe, final_cov, picked, headers, est_used)

        # P6 — S2 band: the opt-in reliable containment scorer judges the texts that WILL be
        # served, only inside the ambiguous band (packing is deterministic given the rows, so
        # the payload is computed first and the scorer sees exactly what serves).
        if (self._coverage_scorer is not None
                and self._coverage_floor <= final_cov < self._coverage_ceiling):
            final_cov = self._coverage_scorer(query, {"claims": list(served_texts)})

        # P7 — the RAG floor: under-coverage appends raw, deduplicated, budget-capped but never
        # empty. Raw priority: (a) the P5 probe chunks; (b) a P1 reuse-stale rebuild's retrieved
        # chunks; (c) else retrieve NOW — which is then the read's single retrieval (the
        # S2-demoted pure serve keeps the shipped v0.5 hit-path contract).
        escalated = False
        raw_chunks: list[Chunk] = []
        if (final_cov < self._coverage_floor
                and self._enable_coverage_escalation
                and self._emits_raw(strat, escalated=True)
                and not self._emits_raw(strat, escalated=False)):
            src = probe_chunks or reuse_evidence
            if not src and not retrieval_ran:
                src = self._retrieve(query, ns)
                retrieval_ran = True
                probe_chunks = src
            raw_chunks = self._floor_pack(src)
            escalated = True
            self._reads_escalated += 1
            if self._residual_spans:
                # Escalation feedback (tier-2 loop): a served raw chunk whose artifact a
                # FRESH unit owns proves that unit under-covered its own source — one lossy
                # signal per owning unit per read (reason "raw_fallback").
                fallback_owners: set[str] = set()
                for c in raw_chunks:
                    for uid in self._artifact_index.get(c.artifact_id, ()):
                        u2 = self._units.get(uid)
                        if u2 is not None and u2.is_fresh and u2.namespace == ns:
                            fallback_owners.add(uid)
                for uid in sorted(fallback_owners):
                    self._bump_lossy(self._units[uid], "raw_fallback")
        needs_retrieval = force_needs_retrieval or final_cov < self._coverage_floor

        # Adaptive-gate recalibration: at the END of a build read (LLM latency hides it; the
        # in-flight read used the previous ceiling).
        if synthesis_ran and self._serve_gate is None and (
                self._pool_noise_ceiling == 0.0 or self._builds_since_calib >= 16):
            self._pool_noise_ceiling = self._pool_noise_floor()
            self._builds_since_calib = 0

        # P8 — PACK & SERVE: render, account, emit, return.
        rendered = (self._pool_render_with_spans(picked, headers, served_spans)
                    if served_spans else self._pool_render(picked, headers))
        pool_claims = [RecalledClaim(claim=h.text, score=h.score, unit_id=h.unit_id)
                       for h in picked]
        top_owner = picked[0].unit_id if picked else ""
        top_unit = self._units.get(top_owner) if top_owner else None

        cache_hit = not synthesis_ran   # True iff ZERO synthesis calls ran this read
        if cache_hit:
            self._reads_hit += 1
            if retrieval_ran:
                self._probe_reads += 1   # retrieved without building — the gate-miss dashboard
            if top_unit is not None:
                # Conservative, continuous-with-v0.5 savings: credit the TOP served owner only.
                self._tokens_saved += top_unit.synth_tokens
        if picked:
            owner_ids = {h.unit_id for h in picked}
            self._pool_units_sum += len(owner_ids)
            self._pool_payloads += 1
            if cache_hit:
                now = self._clock()
                ages = [max(now - owner.freshness_epoch, 0.0)
                        for uid in owner_ids
                        if (owner := self._units.get(uid)) is not None]
                if ages:   # pool mode: age-at-serve = MAX served-owner age this read
                    age = max(ages)
                    self._age_serve_sum += age
                    self._age_serve_max = max(self._age_serve_max, age)
                    self._age_serve_n += 1
        self._emit(
            "pool_served",
            n_claims=len(picked),
            n_units=len({h.unit_id for h in picked}),
            coverage=round(final_cov, 4),
            gate=round(gate, 4),
            est_tokens=est_used,
            budget=self._serve_budget,
            candidates_raw=len(fresh_rows),
            candidates_unique=len(kept),
            reranked=reranked,
            rerank_ms=rerank_ms,
            reuse=forced_serve,
        )

        context: dict[str, Any] = {
            "pool": rendered,
            "serve": "pool",
            "understanding": {"claims": list(served_texts)},
        }
        if escalated:
            context["raw"] = self._attributed_raw(raw_chunks)
        elif strat == ContextStrategy.CONTEXT_RAW and retrieval_ran:
            context["raw"] = self._attributed_raw(probe_chunks or reuse_evidence)
        evidence = (list(probe_chunks or reuse_evidence)
                    if (synthesis_ran or escalated) else [])

        if self._residual_spans:
            self._log_read(
                read_id, qe, ns,
                tuple(sorted({h.unit_id for h in picked} | {u for u, _ in served_spans})),
                tuple(t for _u, t in served_spans),
                est_used,
            )
        return Result(
            understanding={"claims": list(served_texts)},   # no summary key (documented)
            evidence=evidence,
            cache_hit=cache_hit,
            unit_id=top_owner,          # owner of the top-ranked served claim
            confidence=cov0,            # pre-decision pool coverage — what the gate read
            namespace=ns,
            related=(self._related(top_unit, qe, related, ns)
                     if top_unit is not None else []),
            context=context,
            usage=read_usage,
            coverage=final_cov,         # final post-build / post-S2
            escalated=escalated,
            recalled=[],                # unit-path field, kept one version
            pool=pool_claims,
            needs_retrieval=needs_retrieval,
            read_id=read_id,
        )

    def _effective_pool_gate(self) -> float:
        """The serve gate actually applied. An explicit ``serve_gate`` float is ABSOLUTE
        (no adaptation — exact, reproducible benches). None adapts against the pool's own
        null-shaped noise ceiling, clamped to ``cov_default + 0.27`` so a dense near-duplicate
        pool can never gate real answers out (spec §2.3)."""
        if self._serve_gate is not None:
            return self._serve_gate
        _, cov_default = default_thresholds_for(self._embedder)
        return min(max(cov_default, self._pool_noise_ceiling + _POOL_GATE_MARGIN),
                   cov_default + _POOL_GATE_CEILING_MARGIN)

    def _pool_noise_floor(self) -> float:
        """p95 of PROVENANCE-DISJOINT max claim cosines over <=24 fixed-seed probe units —
        null-shaped noise, NOT cross-unit signal: a probe is compared only against atomic rows
        of same-namespace units sharing NO artifact with it, so legitimate same-story signal
        (the whole pool thesis) is excluded from "noise". <8 units -> 0.0. Pure-python path is
        frozen (0.0 -> cov_default gate) above _PUREPY_POOL_WARN_ROWS rows."""
        import random as _random

        units = [u for u in self._units.values() if u.is_fresh and u.query_embedding]
        if len(units) < 8:
            return 0.0
        # Pre-group candidate rows per namespace with owner provenance for the disjoint filter.
        by_ns: dict[str, list[tuple[str, frozenset[str], Any]]] = {}
        total_rows = 0
        for u in self._units.values():
            if not u.is_fresh:
                continue
            texts, embs = self._atomic_rows(u)
            rows = [e for t, e in zip(texts, embs) if t and any(e)]
            if not rows:
                continue
            total_rows += len(rows)
            mat: Any = rows
            if _np is not None:
                m: Any = _np.asarray(rows, dtype=_np.float64)
                n = _np.linalg.norm(m, axis=1, keepdims=True)
                mat = m / _np.where(n > 0, n, 1.0)
            by_ns.setdefault(u.namespace, []).append(
                (u.id, u.provenance.artifact_ids(), mat))
        if _np is None and total_rows > _PUREPY_POOL_WARN_ROWS:
            return 0.0
        sample = _random.Random(len(units)).sample(units, min(24, len(units)))
        maxima: list[float] = []
        for probe in sample:
            arts = probe.provenance.artifact_ids()
            best = 0.0
            if _np is not None:
                q: Any = _np.asarray(probe.query_embedding, dtype=_np.float64)
                qn = float(_np.linalg.norm(q))
                q = q / (qn if qn else 1.0)
            for uid, o_arts, mat in by_ns.get(probe.namespace, []):
                if uid == probe.id or (arts & o_arts):
                    continue   # provenance-disjoint only — same-story units are SIGNAL
                if _np is not None:
                    if mat.shape[1] == len(probe.query_embedding):
                        best = max(best, float((mat @ q).max()))
                else:
                    for e in mat:
                        best = max(best, cosine(probe.query_embedding, e))
            maxima.append(best)
        maxima.sort()
        return maxima[int(0.95 * (len(maxima) - 1))]

    @staticmethod
    def _atomic_rows(unit: Cognition) -> tuple[list[str], list[tuple[float, ...]]]:
        """The unit's ATOMIC pool rows: claims + residuals, summary EXCLUDED (the measured
        hollow-unit poison guard). Positions are preserved (blanks kept as "") so a row's
        ``claim_idx`` always indexes straight into ``unit.claim_embeddings``."""
        claims = unit.understanding.get("claims")
        if not isinstance(claims, list):
            return [], []
        ces = unit.claim_embeddings or ()
        n = min(len(claims), len(ces))
        texts: list[str] = []
        embs: list[tuple[float, ...]] = []
        for i in range(n):
            t = claims[i]
            s = (str(t.get("claim") or t.get("text") or t) if isinstance(t, dict)
                 else str(t)).strip()
            texts.append(s)
            embs.append(tuple(ces[i]))
        return texts, embs

    def _index_unit_rows(self, unit: Cognition) -> None:
        """Push a unit's atomic rows into its namespace claim index (pool mode only) —
        incremental per-unit REPLACE, never an epoch-triggered wholesale rebuild."""
        if self._read_path != "pool":
            return
        texts, embs = self._atomic_rows(unit)
        idx = self._resolve_claim_index(unit.namespace)
        if any(t and any(e) for t, e in zip(texts, embs)):
            idx.add(unit.id, texts, embs)
        else:
            idx.remove(unit.id)

    def _pool_index(self, ns: str) -> ClaimIndex:
        """Resolve + LAZILY hydrate the namespace's claim index on first search — no eager
        O(pool) constructor work; store-loaded units are indexed here once."""
        idx = self._resolve_claim_index(ns)
        if ns not in self._pool_hydrated:
            self._pool_hydrated.add(ns)
            for unit in self._units.values():
                if unit.namespace != ns:
                    continue
                texts, embs = self._atomic_rows(unit)
                if any(t and any(e) for t, e in zip(texts, embs)):
                    idx.add(unit.id, texts, embs)
                    if not unit.is_fresh:
                        idx.mask(unit.id)
        return idx

    def _index_search(
        self, index: ClaimIndex, qe: tuple[float, ...], top_n: int
    ) -> list[tuple[float, ClaimRef, bool]]:
        """One pool scan. ``include_stale`` is telemetry-only and MAY be unsupported by an
        adapter (fresh-only fallback). A broken BYO index fails OPEN — empty pool, so the read
        falls to build/floor: more retrieval, never stale."""
        try:
            try:
                return list(index.search(qe, top_n, include_stale=True))
            except TypeError:
                return list(index.search(qe, top_n))
        except Exception:  # noqa: BLE001 — fail open, never stale, never crash the read
            logger.warning("claim index search failed — treating the pool as empty",
                           exc_info=True)
            return []

    def _pool_ttl(
        self, fresh_rows: list[tuple[float, ClaimRef]]
    ) -> tuple[set[str], bool]:
        """P3 — bounded TTL over the CANDIDATE HEAD only (enough ranked rows to fill
        2x serve_budget, <=32 distinct owners; the whole-store per-read sweep is the
        GraphRAG tax). Returns (owners masked expired-unverified THIS read, any-flip)."""
        policy = self._freshness
        assert policy is not None and policy.max_age is not None
        head: list[str] = []
        seen: set[str] = set()
        budget = 2 * self._serve_budget
        used = 0
        for _score, ref in fresh_rows:
            if ref.unit_id not in seen:
                seen.add(ref.unit_id)
                head.append(ref.unit_id)
                if len(head) >= 32:
                    break
            used += _est_tokens(ref.text)
            if used >= budget:
                break
        now = self._clock()
        masked: set[str] = set()
        flipped = False
        calls = 0
        for uid in head:   # serving-rank priority
            unit = self._units.get(uid)
            if unit is None or not unit.is_fresh:
                continue
            if now - unit.freshness_epoch <= policy.max_age:
                continue
            if policy.revalidate is None:
                self._dirty(unit)   # no revalidator -> conservatively rebuild on read
                self._persist(unit)
                flipped = True
                continue
            if calls >= _MAX_TTL_REVALIDATIONS_PER_READ:
                masked.add(uid)     # expired-unverified for THIS read; later reads revalidate
                continue
            changed, spent = self._ttl_revalidate(
                unit, policy.revalidate, _MAX_TTL_REVALIDATIONS_PER_READ - calls)
            calls += spent
            if changed is None:
                masked.add(uid)     # ran out of revalidation budget mid-owner
            elif changed:
                self._dirty(unit)
                self._persist(unit)
                flipped = True
            else:
                self._mark_fresh(unit)   # revalidate-unchanged: fresh again, NO epoch bump
                self._persist(unit)
        return masked, flipped

    def _ttl_revalidate(
        self,
        unit: Cognition,
        revalidate: Callable[[str], tuple[str, str] | None],
        calls_left: int,
    ) -> tuple[bool | None, int]:
        """Hash-revalidate one owner's artifacts under a call budget. Returns
        (changed | None when the budget ran out unproven, revalidator calls spent)."""
        spent = 0
        for artifact_id in sorted(unit.provenance.artifact_ids()):
            if spent >= calls_left:
                return None, spent
            try:
                fetched = revalidate(artifact_id)
            except Exception:  # revalidation must never crash a read
                fetched = None
            spent += 1
            if fetched is None:
                continue
            text, version = fetched
            event = ChangeEvent(
                artifact_id=artifact_id,
                version=version,
                content_hash=SourceSpan.from_text(artifact_id, text).content_hash,
            )
            if self._content_changed(unit, event):
                return True, spent
        return False, spent

    def _pool_probe_build(
        self,
        query: str,
        qe: tuple[float, ...],
        ns: str,
        chunks: list[Chunk],
        cov0: float,
        gate: float,
    ) -> tuple[Usage | None, bool, set[str]]:
        """P5 probe-classify: group the probe by artifact; CONTAINED groups (every chunk
        retained verbatim by a same-namespace unit — including a ``_synthesis_failed`` one,
        whose retry trigger is provenance/TTL, never every read) cost nothing; STALE-OWNED /
        THIN groups rebuild their existing unit IN PLACE; NOVEL groups build source-anchored
        units. Capped at _MAX_BUILDS_PER_READ synthesis ops, best max-chunk-cosine first.
        Returns (summed usage, any-synthesis-ran, unit ids touched)."""
        groups: dict[str, list[Chunk]] = {}
        for c in chunks:
            groups.setdefault(c.artifact_id, []).append(c)
        work: list[tuple[str, list[Chunk], Cognition | None]] = []
        for aid, chs in groups.items():
            # "" groups under the sentinel key and is always treated NOVEL.
            if aid and all(self._chunk_contained(c, ns) for c in chs):
                # CONTAINED — provenance PROVES coverage... unless the containing owner is
                # lossy-marked (tier-2 loop): then this probe IS the unit's repair touch.
                lossy_owner: Cognition | None = None
                if self._residual_spans:
                    lossy_owner = next(
                        (u for uid in sorted(self._artifact_index.get(aid, ()))
                         if (u := self._units.get(uid)) is not None
                         and u.namespace == ns and u.is_fresh and u.lossy),
                        None)
                if lossy_owner is None:
                    continue
                work.append((aid, chs, lossy_owner))
                continue
            owner: Cognition | None = None
            if aid:
                ns_owners = [u for uid in sorted(self._artifact_index.get(aid, ()))
                             if (u := self._units.get(uid)) is not None
                             and u.namespace == ns]   # D3: containment/ownership is ns-scoped
                stale_owners = [u for u in ns_owners if not u.is_fresh]
                owner = stale_owners[0] if stale_owners else (
                    ns_owners[0] if ns_owners else None)
            work.append((aid, chs, owner))
        if not work:
            if groups:   # ALL CONTAINED + non-empty probe -> structural admission reuse
                self._admission_reuses += 1
                self._emit("admission_reuse", probed_sources=len(groups))
            return None, False, set()
        if len(work) > _MAX_BUILDS_PER_READ:
            # Best max-chunk-cosine artifacts first; the rest converge on later reads.
            def _group_score(chs: list[Chunk]) -> float:
                embs = embed_texts(self._embedder, [c.text for c in chs])
                return max((cosine(qe, tuple(e)) for e in embs if e), default=0.0)

            work.sort(key=lambda w: -_group_score(w[1]))
        usage: Usage | None = None
        touched: set[str] = set()
        built_new = 0
        artifacts: list[str] = []
        for aid, chs, owner in work[:_MAX_BUILDS_PER_READ]:
            built_chunks = self._widen_group(aid, chs) if aid else chs
            if owner is not None and self._residual_spans and owner.lossy:
                # Lossy repair on touch (tier-2 loop): append-only when the source hash is
                # unchanged (old claims never lost), full replace when it changed —
                # unit_repaired fires inside; stale/thin events stay reserved for their paths.
                if not owner.is_fresh:
                    self._stale_reads_prevented += 1
                    self._emit(
                        "stale_read_prevented",
                        unit_id=owner.id,
                        query=query,
                        unit_age_s=round(
                            max(self._clock() - owner.freshness_epoch, 0.0), 3),
                    )
                usage = self._sum_usage(
                    usage, self._repair_unit(owner, query, qe, built_chunks))
                self._rebuilds_by_read += 1
                touched.add(owner.id)
            elif owner is not None:
                reason = "thin" if owner.is_fresh else "stale"
                if reason == "stale":
                    self._stale_reads_prevented += 1
                    self._emit(
                        "stale_read_prevented",
                        unit_id=owner.id,
                        query=query,
                        unit_age_s=round(
                            max(self._clock() - owner.freshness_epoch, 0.0), 3),
                    )
                usage = self._sum_usage(
                    usage, self._build_unit(owner, query, qe, built_chunks))
                self._emit("rebuild_triggered_by_read", unit_id=owner.id,
                           artifact_id=aid, reason=reason)
                self._rebuilds_by_read += 1
                touched.add(owner.id)
            else:
                # NOVEL: source-anchored identity — one unit per (namespace, artifact), always.
                key = hashlib.sha1(f"{ns}|artifact:{aid}".encode("utf-8")).hexdigest()[:16]
                unit = self._units.get(f"cog:{key}")
                if unit is None:
                    unit = Cognition(
                        id=f"cog:{key}",
                        namespace=ns,
                        query=query,            # the triggering query is the birth seed
                        query_embedding=qe,
                        understanding={},
                        evidence=(),
                        provenance=ProvenanceManifest("none", "none"),
                    )
                    self._units[unit.id] = unit
                usage = self._sum_usage(
                    usage, self._build_unit(unit, query, qe, built_chunks))
                self._emit("unit_built", unit_id=unit.id, reason="miss")
                built_new += 1
                touched.add(unit.id)
            artifacts.append(aid)
        self._builds_by_gap += built_new
        self._emit("build_triggered_by_gap", coverage=round(cov0, 4), gate=round(gate, 4),
                   artifacts=sorted(artifacts), n_units_built=built_new)
        return usage, True, touched

    # ------------------------------------------------ pool serving stack (v0.6, spec §3.1)
    def _pool_candidates(
        self, fresh_rows: list[tuple[float, ClaimRef]]
    ) -> list[_PoolHit]:
        """Stage-1: greedy near-dup collapse in descending-cosine order (numpy: vector dedup
        at _POOL_DEDUP_SIM; pure-python: EXACT TEXT only), truncated to _POOL_STAGE1_WIDTH
        unique rows. Collapse is WITHIN-OWNER ONLY (measured finding): a unit's own
        rephrasings/duplicates are redundancy, but cross-owner near-duplicates — even exact
        text — are CORROBORATION, and per-source questions need the owner's own attributed
        copy, so they all survive. Rows whose owner no longer exists are dropped (deleted
        sources never resurrect). ``collapsed_into`` is recorded for gold-transfer."""
        hits: list[_PoolHit] = []
        embs: list[tuple[float, ...] | None] = []
        for score, ref in fresh_rows:
            unit = self._units.get(ref.unit_id)
            if unit is None:
                continue
            ces = unit.claim_embeddings or ()
            emb = (tuple(ces[ref.claim_idx])
                   if 0 <= ref.claim_idx < len(ces) and ces[ref.claim_idx] else None)
            hits.append(_PoolHit(float(score), ref.unit_id, ref.claim_idx, ref.text))
            embs.append(emb)
        collapsed: dict[tuple[str, int], tuple[str, int]] = {}
        kept: list[_PoolHit] = []
        seen_text: dict[tuple[str, str], _PoolHit] = {}   # (owner, text) — never cross-owner
        if _np is not None and hits:
            dim = next((len(e) for e in embs if e), 0)
            mat = _np.empty((min(len(hits), _POOL_STAGE1_WIDTH), dim or 1))
            mat_owner: list[_PoolHit] = []   # kept rows that carry a vector, matrix-parallel
            for h, e in zip(hits, embs):
                if len(kept) >= _POOL_STAGE1_WIDTH:
                    break
                prior = seen_text.get((h.unit_id, h.text))
                if prior is not None:
                    collapsed[(h.unit_id, h.claim_idx)] = (prior.unit_id, prior.claim_idx)
                    continue
                if e is not None and len(e) == dim and mat_owner:
                    v: Any = _np.asarray(e, dtype=_np.float64)
                    n = float(_np.linalg.norm(v))
                    v = v / (n if n else 1.0)
                    sims = mat[: len(mat_owner)] @ v
                    near = next(
                        (mat_owner[j] for j in range(len(mat_owner))
                         if float(sims[j]) >= _POOL_DEDUP_SIM
                         and mat_owner[j].unit_id == h.unit_id),   # same owner ONLY
                        None)
                    if near is not None:
                        collapsed[(h.unit_id, h.claim_idx)] = (near.unit_id, near.claim_idx)
                        continue
                if e is not None and len(e) == dim and len(mat_owner) < mat.shape[0]:
                    v = _np.asarray(e, dtype=_np.float64)
                    n = float(_np.linalg.norm(v))
                    mat[len(mat_owner)] = v / (n if n else 1.0)
                    mat_owner.append(h)
                seen_text[(h.unit_id, h.text)] = h
                kept.append(h)
        else:
            for h in hits:
                if len(kept) >= _POOL_STAGE1_WIDTH:
                    break
                prior = seen_text.get((h.unit_id, h.text))
                if prior is not None:
                    collapsed[(h.unit_id, h.claim_idx)] = (prior.unit_id, prior.claim_idx)
                    continue
                seen_text[(h.unit_id, h.text)] = h
                kept.append(h)
        self._last_collapsed = collapsed
        return kept

    def _pool_rank(
        self, query: str, cand: list[_PoolHit]
    ) -> tuple[list[_PoolHit], bool, float | None]:
        """Stage-2: the BYO rerank hook — SERVING ORDER only. Sorted by (-score,
        cosine_position), stable and deterministic; NaN sorts as -inf. Any exception,
        length mismatch, or non-float score degrades to cosine order + ``rerank_failed``."""
        if self._reranker is None or not cand:
            return cand, False, None
        t0 = time.perf_counter()
        try:
            raw = list(self._reranker(query, [h.text for h in cand]))
            if len(raw) != len(cand):
                raise ValueError(
                    f"reranker returned {len(raw)} scores for {len(cand)} texts")
            scores: list[float] = []
            for s in raw:
                if isinstance(s, bool) or not isinstance(s, (int, float)):
                    raise TypeError(f"non-float reranker score: {s!r}")
                f = float(s)
                scores.append(float("-inf") if math.isnan(f) else f)
        except Exception as exc:  # noqa: BLE001 — a BYO reranker must never break serving
            self._emit("rerank_failed", reason=str(exc))
            return cand, False, round((time.perf_counter() - t0) * 1000.0, 3)
        ms = round((time.perf_counter() - t0) * 1000.0, 3)
        order = sorted(range(len(cand)), key=lambda i: (-scores[i], i))
        return [cand[i] for i in order], True, ms

    def _pool_pack(
        self, order: list[_PoolHit]
    ) -> tuple[list[_PoolHit], int, dict[str, str]]:
        """Stage-3, THE packing contract: est(s) = max(1, len(s)//4); headers and separators
        COUNT against serve_budget; walk the final order and STOP at the first overflow — no
        skip-scanning (rank order is the contract). AT-LEAST-ONE: an oversized first claim
        serves with its header dropped, and if still over, serves anyway + overrun event."""
        budget = self._serve_budget
        picked: list[_PoolHit] = []
        headers: dict[str, str] = {}
        running = 0
        for hit in order:
            opening = hit.unit_id not in headers
            header = self._pool_header_text(hit.unit_id) if opening else ""
            cost = _est_tokens("- " + hit.text)
            if opening:
                cost += 1   # the group separator/open
                if header:
                    cost += _est_tokens(header) + 1
            if picked:
                if running + cost > budget:
                    break   # STOP — no skip-scanning
            else:
                if cost > budget and header:
                    header = ""   # AT-LEAST-ONE: drop the header first
                    cost = _est_tokens("- " + hit.text) + 1
                if cost > budget:
                    self._emit("pool_budget_overrun", est_tokens=cost)
            if opening:
                headers[hit.unit_id] = header
            picked.append(hit)
            running += cost
        return picked, running, headers

    @staticmethod
    def _default_header(unit: Cognition) -> str:
        """The built-in attribution line (measured finding: an unattributed pool payload
        makes per-source questions unanswerable BY CONSTRUCTION — claims cannot name their
        own outlet). Default = the spec §7.1 fallback header, shipped after the fork's
        paid confirm measured the opaque id form 0.089 BELOW the rig-parity header
        (0.6413 vs 0.7306 strict, n=605 — far past the 0.03 fork criterion): the unit's
        build QUERY carries source-identifying text (e.g. "key facts of the article:
        {title}"), so serve its first 60 chars as a "## " heading. Only a unit with no
        query text falls back to the opaque "[source: {artifact_id}]" line; an explicit
        ``pool_header`` callable still overrides everything."""
        q = (unit.query or "").strip()
        if q:
            return "## " + q[:60]
        aid = unit.evidence[0].artifact_id if unit.evidence else ""
        return f"[source: {aid or unit.id}]"

    def _pool_header_text(self, unit_id: str) -> str:
        """The per-group attribution header. ``pool_header=None`` falls back to the built-in
        default (``## {unit.query[:60]}``, else ``[source: ...]`` — spec §7.1 fork) — the
        pool path is UNABLE to serve an unattributed group; a caller callback overrides it.
        A raising hook is swallowed, never breaks serving (headerless then, and its cost is
        never counted — the shipped contract)."""
        unit = self._units.get(unit_id)
        if unit is None:
            return ""
        if self._pool_header is None:
            return self._default_header(unit)
        try:
            return str(self._pool_header(unit) or "")
        except Exception:  # noqa: BLE001 — a header hook must never break serving
            logger.debug("pool_header raised — ignored", exc_info=True)
            return ""

    @staticmethod
    def _pool_render(picked: list[_PoolHit], headers: dict[str, str]) -> str:
        """Stage-4: group by owner; group order = ascending best serving-rank in group;
        WITHIN-group order = serving order (the measured operating point). Header once per
        group, "- " claim lines, groups joined by blank lines."""
        order_of: list[str] = []
        groups: dict[str, list[str]] = {}
        for hit in picked:
            if hit.unit_id not in groups:
                groups[hit.unit_id] = []
                order_of.append(hit.unit_id)
            groups[hit.unit_id].append(hit.text)
        parts: list[str] = []
        for uid in order_of:
            head = headers.get(uid, "")
            body = "\n".join("- " + c for c in groups[uid])
            parts.append((head + "\n" + body) if head else body)
        return "\n\n".join(parts)

    # ------------------------------------ tier-2 span side channel (v0.6, opt-in serving)
    def _residual_span_rows(
        self, ns: str
    ) -> tuple[list[tuple[str, ResidualSpan]], Any]:
        """(owner_id, span) rows over FRESH units in ``ns`` plus a pre-normalized numpy
        matrix (None on the pure path) — built lazily and keyed on
        ``(ns, _rows_epoch, _status_gen)``, epoch-invalidated exactly like the pool index,
        so a rebuild or a freshness flip refreshes it on the next read."""
        marker = (ns, self._rows_epoch, self._status_gen)
        state = self._span_state
        if state is not None and state[0] == marker:
            return state[1], state[2]
        self._hydrate_span_embeddings(ns)
        rows: list[tuple[str, ResidualSpan]] = []
        dim = 0
        for uid, u in self._units.items():
            if u.namespace != ns or not u.is_fresh or not u.residual_spans:
                continue
            for span in u.residual_spans:
                if not span.embedding or not any(span.embedding):
                    continue
                if dim == 0:
                    dim = len(span.embedding)
                if len(span.embedding) == dim:   # embedder-swap leftovers can't poison the matrix
                    rows.append((uid, span))
        matrix: Any = None
        if _np is not None and rows:
            m: Any = _np.asarray([s.embedding for _, s in rows], dtype=_np.float64)
            n = _np.linalg.norm(m, axis=1, keepdims=True)
            matrix = m / _np.where(n > 0, n, 1.0)
        self._span_state = (marker, rows, matrix)
        return rows, matrix

    @staticmethod
    def _score_vectors(
        matrix: Any, embs: list[tuple[float, ...]], qe: tuple[float, ...]
    ) -> list[float]:
        """Query cosine per stored vector — one matvec on the numpy path, the exact
        pure-python twin otherwise. Shared by the span side channel and the key overlay."""
        if matrix is not None:
            q: Any = _np.asarray(qe, dtype=_np.float64)
            n = float(_np.linalg.norm(q))
            q = q / (n if n else 1.0)
            return [float(s) for s in (matrix @ q).tolist()]
        return [cosine(qe, e) for e in embs]

    @staticmethod
    def _score_spans(
        rows: list[tuple[str, ResidualSpan]], matrix: Any, qe: tuple[float, ...]
    ) -> list[float]:
        """Query cosine per span row — the anomaly channel's and report_refusal's scorer."""
        return SemanticCache._score_vectors(matrix, [s.embedding for _, s in rows], qe)

    def _log_read(
        self,
        read_id: str,
        qe: tuple[float, ...],
        ns: str,
        served_units: tuple[str, ...],
        served_span_texts: tuple[str, ...],
        est_used: int,
    ) -> None:
        """Remember this read in the fallback ring buffer (last _READ_LOG_CAP reads):
        what was asked (the query embedding), where (ns), and what served — so a later
        report_refusal(read_id) can score the namespace's residual spans against the
        ORIGINAL question and skip spans the refused payload already contained."""
        self._read_log[read_id] = (qe, ns, served_units, served_span_texts,
                                   est_used, self._serve_budget)
        while len(self._read_log) > _READ_LOG_CAP:
            expired = next(iter(self._read_log))
            self._read_log.pop(expired)
            self._expire_provisional_keys(expired)   # unconfirmed keys expire with the ring

    def _hydrate_span_embeddings(self, ns: str) -> None:
        """Store-loaded spans carry NO embeddings (serde persists text + provenance only —
        the embedding is regenerable, and ~30KB/span of JSON is not). Re-embed every missing
        one in ONE batch on the namespace's first side-channel use; freshly-built spans keep
        their in-memory embedding, so nothing re-embeds twice in-process."""
        units = [u for u in self._units.values()
                 if u.namespace == ns and u.is_fresh
                 and any(not s.embedding for s in u.residual_spans)]
        if not units:
            return
        texts = [s.text for u in units for s in u.residual_spans if not s.embedding]
        embs = iter(embed_texts(self._embedder, texts))
        for u in units:
            u.residual_spans = tuple(
                s if s.embedding else ResidualSpan(
                    text=s.text, artifact_id=s.artifact_id, chunk_idx=s.chunk_idx,
                    embedding=tuple(next(embs)))
                for s in u.residual_spans
            )

    # --------------------------------------- query keys (v0.6, opt-in query_keys)
    def _attach_key(
        self, unit: Cognition, qe: tuple[float, ...], span_text: str, read_id: str
    ) -> None:
        """Attach one PROVISIONAL query key (optimistic-attach: it retrieves immediately,
        no confirmation wait). Cap: at most _MAX_KEYS_PER_UNIT keys per unit — the newcomer
        always lands; the lowest-hits EXISTING key (ties: oldest) is evicted instead."""
        texts, _ = self._atomic_rows(unit)
        claim_idx = next((i for i, t in enumerate(texts) if t == span_text), -1)
        keys = list(unit.query_keys)
        keys.append(QueryKey(embedding=qe, claim_idx=claim_idx,
                             span_text=span_text, hits=0, read_id=read_id))
        if len(keys) > _MAX_KEYS_PER_UNIT:
            evict = min(range(len(keys) - 1), key=lambda i: (keys[i].hits, i))
            keys.pop(evict)
        unit.query_keys = tuple(keys)
        self._keys_by_read.setdefault(read_id, []).append(unit.id)
        self._key_gen += 1
        self._emit("key_attached", unit_id=unit.id, provisional=True)

    def _expire_provisional_keys(self, read_id: str) -> None:
        """A read aged out of the ring: its never-confirmed keys go with it (the confirmed
        ones — read_id already cleared — stay durable on the unit)."""
        for uid in self._keys_by_read.pop(read_id, []):
            unit = self._units.get(uid)
            if unit is None:
                continue
            kept = tuple(k for k in unit.query_keys if k.read_id != read_id)
            if len(kept) != len(unit.query_keys):
                unit.query_keys = kept
                self._key_gen += 1

    def _rebind_keys(self, unit: Cognition) -> None:
        """Re-point each key at its claim row by the promoted fact's TEXT identity (claim
        positions regenerate on every rebuild; the text is the durable binding). A key whose
        fact is not (yet) a verbatim claim stays span-level (claim_idx -1, inert as a row)."""
        if not unit.query_keys:
            return
        texts, _ = self._atomic_rows(unit)
        pos = {t: i for i, t in enumerate(texts) if t}
        for key in unit.query_keys:
            key.claim_idx = pos.get(key.span_text, -1)

    def _query_key_rows(
        self, ns: str
    ) -> tuple[list[tuple[str, int, str, QueryKey]], Any]:
        """(owner_id, row_idx, row_text, key) rows over FRESH units' keys (provisional AND
        confirmed — optimistic-attach serves immediately) plus the numpy matrix — lazily
        built, keyed like the span state with ``_key_gen`` added so an attach/expiry/
        eviction refreshes it without a rows-epoch bump.

        A claim-bound key (claim_idx >= 0) serves its CLAIM text. A span-level key
        (claim_idx == -1 — the promoted fact is not yet a claim, which is the NORMAL state
        for a behaviorally-attached key, since spans are captured precisely because their
        facts are absent from every claim) serves its own SPAN TEXT: that sentence is
        exactly the content that rescued the verified refusal round-trip, so serving it on
        key fire is serving behavior-verified content. Span rows get DISTINCT negative
        sentinel indices per unit so multiple span-level keys never collide on the
        (unit, idx) identity downstream."""
        marker = (ns, self._rows_epoch, self._status_gen, self._key_gen)
        state = self._key_state
        if state is not None and state[0] == marker:
            return state[1], state[2]
        rows: list[tuple[str, int, str, QueryKey]] = []
        dim = 0
        for uid, u in self._units.items():
            if u.namespace != ns or not u.is_fresh or not u.query_keys:
                continue
            texts, _embs = self._atomic_rows(u)
            sentinel = 0                         # -1, -2, ... per span-level key of this unit
            for key in u.query_keys:
                if not key.embedding or not any(key.embedding):
                    continue
                if 0 <= key.claim_idx < len(texts) and texts[key.claim_idx]:
                    row_idx, row_text = key.claim_idx, texts[key.claim_idx]
                elif key.span_text:
                    sentinel -= 1
                    row_idx, row_text = sentinel, key.span_text
                else:
                    continue
                if dim == 0:
                    dim = len(key.embedding)
                if len(key.embedding) == dim:
                    rows.append((uid, row_idx, row_text, key))
        matrix: Any = None
        if _np is not None and rows:
            m: Any = _np.asarray([k.embedding for _, _, _, k in rows], dtype=_np.float64)
            n = _np.linalg.norm(m, axis=1, keepdims=True)
            matrix = m / _np.where(n > 0, n, 1.0)
        self._key_state = (marker, rows, matrix)
        return rows, matrix

    def _apply_query_keys(
        self,
        ns: str,
        qe: tuple[float, ...],
        fresh_rows: list[tuple[float, ClaimRef]],
        masked: set[str],
        fired: set[int],
    ) -> list[tuple[float, ClaimRef]]:
        """The key overlay on the pool scan (the MATCH RULE): a key row's similarity counts
        ONLY at/above ``key_floor`` — below it the row is ignored entirely, so keys never
        compete at low similarity (the v0.2-regression guard). A claim's serving score
        becomes max(content_sim, qualifying key_sim): its row is raised in place or injected
        when the scan missed it; content rows are never displaced, and dedup never sees a
        separate key row (one row per claim, key-boosted). Idempotent — safe to re-apply
        after a rebuild re-scan; ``fired`` de-dupes key_fired/hits within one read."""
        if not self._query_keys:
            return fresh_rows
        rows, matrix = self._query_key_rows(ns)
        if not rows:
            return fresh_rows
        sims = self._score_vectors(matrix, [k.embedding for _, _, _, k in rows], qe)
        best: dict[tuple[str, int], tuple[float, str, QueryKey]] = {}
        for (uid, cidx, text, key), sim in zip(rows, sims):
            if sim < self._key_floor or uid in masked:
                continue                         # ignored ENTIRELY below the floor
            prior = best.get((uid, cidx))
            if prior is None or sim > prior[0]:
                best[(uid, cidx)] = (sim, text, key)
        if not best:
            return fresh_rows
        out: list[tuple[float, ClaimRef]] = []
        for score, ref in fresh_rows:
            hit = best.pop((ref.unit_id, ref.claim_idx), None)
            if hit is not None and hit[0] > score:
                score = hit[0]                   # raised in place: max(content, key)
                self._fire_key(hit[2], ref.unit_id, hit[0], fired)
            out.append((score, ref))
        for (uid, cidx), (sim, text, key) in best.items():
            out.append((sim, ClaimRef(uid, cidx, text)))   # the scan missed it: inject
            self._fire_key(key, uid, sim, fired)
        out.sort(key=lambda r: (-r[0], r[1].unit_id, r[1].claim_idx))
        return out

    def _fire_key(self, key: QueryKey, unit_id: str, sim: float, fired: set[int]) -> None:
        """One key changed a read's outcome (raised or injected its claim): count the hit
        (eviction keeps proven keys) and emit ``key_fired`` — once per key per read."""
        if id(key) in fired:
            return
        fired.add(id(key))
        key.hits += 1
        self._emit("key_fired", unit_id=unit_id, key_sim=round(sim, 4))

    def _serve_residual_spans(
        self,
        ns: str,
        qe: tuple[float, ...],
        claim_sim: float,
        picked: list[_PoolHit],
        headers: dict[str, str],
        est_used: int,
    ) -> tuple[list[tuple[str, str]], int, dict[str, str]]:
        """The side channel (zero pool crowding): score the fresh units' tier-2 spans against
        the query IN MEMORY (never a retrieval — the single-retrieval invariant holds) and,
        under the RANKING-ANOMALY RULE — span similarity strictly above the best fresh CLAIM
        similarity + ``span_margin`` — append up to ``_SPANS_PER_READ`` labeled spans to the
        served payload, inside the SAME ``serve_budget`` accounting (a new group's header is
        costed like the packer does; a span that does not fit is not served). Every span serve
        emits ``residual_served`` and counts one lossy signal against its owner."""
        rows, matrix = self._residual_span_rows(ns)
        if not rows:
            return [], est_used, headers
        sims = self._score_spans(rows, matrix, qe)
        order = sorted(range(len(rows)),
                       key=lambda i: (-sims[i], rows[i][0], rows[i][1].chunk_idx))
        served: list[tuple[str, str]] = []
        served_texts = {h.text for h in picked}
        out_headers = headers
        opened = set(headers)                 # groups whose open + header are already paid for
        budget = self._serve_budget
        for i in order:
            if len(served) >= _SPANS_PER_READ:
                break
            if sims[i] <= claim_sim + self._span_margin:
                break                         # sorted desc: no anomaly below this point
            uid, span = rows[i]
            if span.text in served_texts:
                continue                      # already served verbatim as a claim
            cost = _est_tokens("- " + _SPAN_LABEL + " " + span.text)
            header = out_headers.get(uid, "")
            if uid not in opened:
                header = self._pool_header_text(uid)
                cost += 1                     # the group separator/open
                if header:
                    cost += _est_tokens(header) + 1
            if est_used + cost > budget:
                break                         # a bonus net never overruns the budget
            if uid not in opened:
                if out_headers is headers:
                    out_headers = dict(headers)
                out_headers[uid] = header
                opened.add(uid)
            est_used += cost
            served.append((uid, span.text))
            served_texts.add(span.text)
            self._emit("residual_served", unit_id=uid,
                       span_sim=round(sims[i], 4), claim_sim=round(claim_sim, 4))
            unit = self._units.get(uid)
            if unit is not None:
                self._bump_lossy(unit, "span_hits")
        return served, est_used, out_headers

    @staticmethod
    def _pool_render_with_spans(
        picked: list[_PoolHit], headers: dict[str, str], spans: list[tuple[str, str]]
    ) -> str:
        """:meth:`_pool_render` plus the side channel: each served span renders as a labeled
        line UNDER its owner's group (attribution preserved); a span whose owner served no
        claims opens its own attributed group at the end."""
        order_of: list[str] = []
        lines: dict[str, list[str]] = {}
        for hit in picked:
            if hit.unit_id not in lines:
                lines[hit.unit_id] = []
                order_of.append(hit.unit_id)
            lines[hit.unit_id].append("- " + hit.text)
        for uid, text in spans:
            if uid not in lines:
                lines[uid] = []
                order_of.append(uid)
            lines[uid].append("- " + _SPAN_LABEL + " " + text)
        parts: list[str] = []
        for uid in order_of:
            head = headers.get(uid, "")
            body = "\n".join(lines[uid])
            parts.append((head + "\n" + body) if head else body)
        return "\n\n".join(parts)

    @staticmethod
    def _attributed_raw(chunks: list[Chunk]) -> list[str]:
        """Raw payload entries with their source attribution line (measured finding:
        escalation raw was as outlet-blind as the pool payload was) — the same
        ``[source: ...]`` format the default pool header uses, straight from each chunk's
        own artifact_id; a chunk with no artifact stays bare."""
        return [f"[source: {c.artifact_id}]\n{c.text}" if c.artifact_id else c.text
                for c in chunks]

    def _floor_pack(self, chunks: list[Chunk]) -> list[Chunk]:
        """P7 raw payload: deduplicated, packed to <= serve_budget est. tokens — but never
        fewer than one chunk (the floor must floor)."""
        out: list[Chunk] = []
        seen: set[tuple[str, str]] = set()
        used = 0
        for c in chunks:
            key = (c.artifact_id, c.text)
            if key in seen:
                continue
            seen.add(key)
            cost = _est_tokens(c.text)
            if out and used + cost > self._serve_budget:
                break
            out.append(c)
            used += cost
        return out

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
            S: Any = _np.asarray([s for s in seeds if len(s) == dim], dtype=_np.float64)
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

    def report_refusal(self, read_id: str) -> str | None:
        """The behavioral residual-span fallback (v0.6, requires ``residual_spans=True``) —
        the PRIMARY tier-2 serving trigger: rank at serve time is a guess, a refusal is
        evidence. When YOUR answerer refuses over a read's served payload, hand back that
        read's ``Result.read_id``; the cache re-scores the namespace's residual spans against
        the ORIGINAL query embedding and, when the best span clears ``span_serve_floor``,
        returns a RETRY PAYLOAD — up to 2 attributed ``[source excerpt]`` lines (spans the
        refused payload already contained are skipped) to append to the original context for
        one retry call. Each returned span counts a lossy signal against its owning unit
        exactly like the anomaly serve path, and emits ``residual_fallback``.

        Returns ``None`` (emitting nothing) when no span qualifies, when the read_id is
        unknown or has aged out of the ring buffer, or when the feature is off — never
        raises. In-memory only: no retrieval, no LLM call."""
        if not self._residual_spans:
            return None
        rec = self._read_log.get(read_id)
        if rec is None:
            return None
        qe, ns, _served_units, served_span_texts, _est_used, _budget = rec
        rows, matrix = self._residual_span_rows(ns)
        if not rows:
            return None
        sims = self._score_spans(rows, matrix, qe)
        order = sorted(range(len(rows)),
                       key=lambda i: (-sims[i], rows[i][0], rows[i][1].chunk_idx))
        chosen: list[tuple[str, str]] = []
        headers: dict[str, str] = {}
        seen = set(served_span_texts)
        for i in order:
            if len(chosen) >= _SPANS_PER_READ:
                break
            if sims[i] < self._span_serve_floor:
                break                     # sorted desc: nothing below clears the floor
            uid, span = rows[i]
            if span.text in seen:
                continue                  # the refused payload already carried it
            seen.add(span.text)
            chosen.append((uid, span.text))
            headers.setdefault(uid, self._pool_header_text(uid))
            self._emit("residual_fallback", read_id=read_id, unit_id=uid,
                       span_sim=round(sims[i], 4))
            unit = self._units.get(uid)
            if unit is not None:
                self._bump_lossy(unit, "span_hits")   # the behavioral residual-hit
                if self._query_keys:
                    # Optimistic-attach: the refused query's embedding becomes a PROVISIONAL
                    # alternate key on the span's owner — retrieving immediately, confirmed
                    # durable by report_success, expiring with the ring otherwise.
                    self._attach_key(unit, qe, span.text, read_id)
        if not chosen:
            return None
        return self._pool_render_with_spans([], headers, chosen)

    def report_success(self, read_id: str) -> None:
        """The symmetric half of the behavioral channel (requires ``query_keys=True``):
        call it when the retry over a :meth:`report_refusal` payload SUCCEEDED. It CONFIRMS
        that read's provisional query keys — they stop expiring with the read ring and
        persist on their units (serde) — and emits ``key_confirmed`` per unit. Unknown or
        expired read_id, or a read that attached no keys: a silent no-op, never raises."""
        if not self._query_keys:
            return
        for uid in self._keys_by_read.pop(read_id, []):
            unit = self._units.get(uid)
            if unit is None:
                continue
            confirmed = False
            for key in unit.query_keys:
                if key.read_id == read_id:
                    key.read_id = ""          # confirmed: durable, ring-expiry-proof
                    confirmed = True
            if confirmed:
                self._emit("key_confirmed", unit_id=uid)
                self._persist(unit)

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
                self._dirty(unit)
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

    def has_source(self, artifact_id: str) -> bool:
        """True when any cached unit's provenance depends on ``artifact_id`` — the cheap
        pre-check for change-feed adapters (e.g. the MCP watched-folder server), which
        would otherwise fire ``source_changed`` for files no unit has ever read and trip
        the loud matched-no-units wiring warning on every edit of an unread file."""
        return artifact_id in self._artifact_index

    def _evict(self, unit_id: str) -> None:
        unit = self._units.pop(unit_id, None)
        for index in (self._artifact_index, self._entity_index):
            for key, unit_ids in list(index.items()):
                unit_ids.discard(unit_id)
                if not unit_ids:
                    del index[key]
        if self._store is not None:
            self._store.delete(unit_id)
        self._rows_epoch += 1     # the unit's claim rows were removed — monotone
        if unit is not None and self._read_path == "pool":
            idx = self._existing_index(unit.namespace)
            if idx is not None:
                idx.remove(unit_id)   # a deleted source's claims must never resurrect

    # --------------------------------------------------- time-based freshness
    def _refresh_if_expired(self, unit: Cognition) -> None:
        """On TTL expiry, revalidate feed-less sources by hash; skip no-op rebuilds."""
        policy = self._freshness
        if policy is None or policy.max_age is None or not unit.is_fresh:
            return
        if self._clock() - unit.freshness_epoch <= policy.max_age:
            return
        if policy.revalidate is None:
            self._dirty(unit)  # no revalidator -> conservatively rebuild on read
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
            self._dirty(unit)
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
        """v0.4 — derive ``coverage_floor`` from a COST trade-off instead of guessing a
        constant: ``tau* = 1 - escalation_cost / miss_cost``. The dearer a WRONG answer
        (``miss_cost``) is relative to one extra retrieval (``escalation_cost``), the higher the
        floor → the more eagerly the cache escalates to the RAG floor. Pass the result as
        ``coverage_floor=``. Clamped to [0, 1]; a non-positive ``miss_cost`` yields 0 (never escalate)."""
        if miss_cost <= 0:
            return 0.0
        return round(max(0.0, min(1.0, 1.0 - escalation_cost / miss_cost)), 4)

    def stats(self) -> dict[str, Any]:
        """Cache size + read-time observability. ``escalation_rate`` is the fraction of
        cache HITS that fell back to fresh raw — a high value means the cached
        understanding under-covers real queries (deepen it, or lower coverage_floor).

        Pool mode (``read_path="pool"``) adds the §4.5 keys — pool sizes/mask rate, the
        effective gate + noise ceiling, and the serve/probe/admission/retrieval counters
        that make gate miscalibration one dashboard number. ``avg/max_age_at_serve_s`` are
        redefined in pool mode as the mean/max over reads of the MAX served-owner age.
        ``tokens_saved`` on a pool hit credits the TOP-ranked served claim's owner only —
        conservative, continuous with v0.5 accounting."""
        hits = self._reads_hit
        synth_tokens = self._synth_prompt_tokens + self._synth_completion_tokens
        avg_synth = synth_tokens / self._synth_calls if self._synth_calls else 0.0
        now = self._clock()
        fresh_ages = [max(now - u.freshness_epoch, 0.0) for u in self._units.values() if u.is_fresh]
        base: dict[str, Any] = {
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
            # Effective operating thresholds/knobs, so the cost/risk point is observable (tau*)
            # and a caller can confirm which additive mechanisms are actually active this run.
            "coverage_floor": self._coverage_floor,
            "recall_threshold": self._effective_recall_threshold(),
            "hit_threshold": self._threshold,
            "hit_margin": self._hit_margin,
            "select_floor": self._select_floor,
            "residual_floor": self._residual_floor,
        }
        if self._read_path == "pool":
            fresh_rows = 0
            total_rows = 0
            for u in self._units.values():
                texts, embs = self._atomic_rows(u)
                n = sum(1 for t, e in zip(texts, embs) if t and any(e))
                total_rows += n
                if u.is_fresh:
                    fresh_rows += n
            base.update({
                "read_path": self._read_path,
                "serve_budget": self._serve_budget,
                "serve_gate": self._serve_gate,
                "serve_gate_effective": round(self._effective_pool_gate(), 4),
                "pool_noise_ceiling": round(self._pool_noise_ceiling, 4),
                "pool_claims_fresh": fresh_rows,
                "pool_claims_total": total_rows,
                "pool_mask_rate": (round(1.0 - fresh_rows / total_rows, 3)
                                   if total_rows else 0.0),
                "pool_serves": self._pool_serves,
                "probe_reads": self._probe_reads,
                "admission_reuses": self._admission_reuses,
                "retrievals": self._retrievals,
                "rebuilds_by_read": self._rebuilds_by_read,
                "builds_by_gap": self._builds_by_gap,
                "avg_units_per_serve": (round(self._pool_units_sum / self._pool_payloads, 2)
                                        if self._pool_payloads else 0.0),
                "pool_scan_slow": self._pool_scan_slow,
            })
        return base
