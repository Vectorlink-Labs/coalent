"""The Cognition unit — understanding + RETAINED raw evidence + provenance.

Addressed by the embedding of the query that built it. It keeps its evidence so
the cache can never return less than plain retrieval.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..domain.models import ProvenanceManifest, Status
from .ports import Chunk


@dataclass(frozen=True, slots=True)
class ResidualSpan:
    """A tier-2 residual span (v0.6, opt-in ``residual_spans``): a fact-bearing source
    sentence the extractor did NOT cover, retained VERBATIM on its unit with provenance
    captured at build (artifact + evidence-chunk index — never re-derived later). Spans
    live on the unit only — they are NEVER rows in the claim pool (news pools stay lean);
    the pool read path serves one as a labeled "[source excerpt]" side channel when it
    outranks every fresh claim for a query (the extraction-loss net)."""

    text: str
    artifact_id: str
    chunk_idx: int                          # index into the unit's evidence at build time
    embedding: tuple[float, ...] = ()       # embedded once at build (batch call), persisted


@dataclass(slots=True)
class QueryKey:
    """A behavioral ALTERNATE retrieval key (v0.6, opt-in ``query_keys``): the embedding of
    a real query the cache failed on, bound to the claim that answers it — attached
    optimistically by ``report_refusal`` (provisional; expires with the read ring),
    confirmed durable by ``report_success``. At read time a key row joins the pool scan for
    its claim and counts ONLY at/above ``key_floor`` similarity (never competes low — the
    regression guard lives in the scoring rule). Keys carry no text: the embedding IS the
    key, so it is persisted (confirmed keys only), capped per unit."""

    embedding: tuple[float, ...]
    claim_idx: int = -1        # target claim row; -1 = still span-level (pre-promotion)
    span_text: str = ""        # the promoted fact's durable identity (rebinds claim_idx)
    hits: int = 0              # times this key fired (eviction keeps the proven ones)
    read_id: str = ""          # non-empty = PROVISIONAL (expires with the read ring)


@dataclass(slots=True)
class Cognition:
    """One cached piece of decision-ready understanding."""

    id: str
    namespace: str
    query: str                              # the query that built it
    query_embedding: tuple[float, ...]      # the cache key (semantic)
    understanding: dict[str, Any]
    evidence: tuple[Chunk, ...]             # retained raw — guarantees the RAG floor
    provenance: ProvenanceManifest          # which sources it depends on (invalidation)
    # --- v0.3: the unit is keyed by what it KNOWS, not the seed query ---
    understanding_embedding: tuple[float, ...] = ()       # embedding of the understanding digest
    claim_embeddings: tuple[tuple[float, ...], ...] = ()  # per-claim embeddings (semantic coverage)
    synth_tokens: int = 0                                 # v0.4: this unit's build cost in tokens —
                                                          # a HIT credits this as saved (reload-safe)
    hit_queries: tuple[str, ...] = ()                     # behavioral: queries that hit this unit
    # --- v0.6 (opt-in residual_spans): the tier-2 extraction-loss net + repair signals ---
    residual_spans: tuple[ResidualSpan, ...] = ()         # uncovered fact sentences (side channel)
    span_hits: int = 0                                    # lossy signals: span serves + raw fallbacks
    lossy: bool = False                                   # marked lossy -> repairs on next touch
    # --- v0.6 (opt-in query_keys): behavioral alternate retrieval keys ---
    query_keys: tuple[QueryKey, ...] = ()                 # confirmed keys persist; provisional
    #                                                       ones expire with the read ring
    status: Status = Status.FRESH
    freshness_epoch: float = field(default_factory=time.time)
    hits: int = 0
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)

    @property
    def is_fresh(self) -> bool:
        return self.status == Status.FRESH

    @property
    def needs_backfill(self) -> bool:
        """A pre-v0.3 unit (no understanding embedding yet) — backfill once on load."""
        return not self.understanding_embedding

    def touch(self, query: str | None = None, *, max_queries: int = 16) -> None:
        self.hits += 1
        self.last_access = time.time()
        # Behavioral seed: remember the queries that actually hit this unit (bounded).
        if query and query not in self.hit_queries:
            self.hit_queries = (self.hit_queries + (query,))[-max_queries:]

    def mark_dirty(self) -> None:
        self.status = Status.DIRTY

    def mark_fresh(self) -> None:
        self.status = Status.FRESH
        self.freshness_epoch = time.time()
