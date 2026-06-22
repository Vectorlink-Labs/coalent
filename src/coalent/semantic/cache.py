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
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..domain.models import ChangeEvent, ProvenanceManifest, SourceSpan
from .embedding import (
    Embedder,
    cosine,
    default_embedder,
    default_thresholds_for,
    embed_texts,
    tokenize,
)
from .ports import Chunk, Retriever, Synthesizer
from .store import CognitionStore
from .unit import Cognition

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Related:
    """A cross-unit relationship surfaced on a read (shared entity or source)."""

    unit_id: str
    understanding: dict[str, Any]
    evidence: list[Chunk]
    relation: str   # "shared_entity" | "shared_source"
    score: float    # relevance of this related unit to the current query


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
    """Embedding-keyed cognitive cache over understanding + raw evidence."""

    def __init__(
        self,
        retriever: Retriever,
        synthesizer: Synthesizer,
        *,
        embedder: Embedder | None = None,
        hit_threshold: float | None = None,
        coverage_floor: float | None = None,
        understanding_weight: float = 0.7,
        route_by_claim: bool = False,
        learn_behavior: bool = True,
        max_hit_queries: int = 16,
        enable_coverage_escalation: bool = True,
        coverage_scorer: Callable[[str, dict[str, Any]], float] | None = None,
        coverage_ceiling: float = 1.0,
        relevance_gate: Callable[[str, list[Chunk]], list[Chunk]] | None = None,
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
        self._coverage_floor = cov_default if coverage_floor is None else coverage_floor
        self._understanding_weight = understanding_weight
        self._route_by_claim = route_by_claim
        self._learn_behavior = learn_behavior
        self._max_hit_queries = max_hit_queries
        self._enable_coverage_escalation = enable_coverage_escalation
        self._coverage_scorer = coverage_scorer
        self._coverage_ceiling = coverage_ceiling
        self._relevance_gate = relevance_gate
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

        best_id, best_score = self._best_match(qe, ns)
        if best_id is not None and best_score >= self._threshold:
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
                self._materialize_into(unit, query, qe, ns)
                cache_hit = False
            confidence = best_score
        else:
            unit = self._new_unit(query, qe, ns)
            self._materialize_into(unit, query, qe, ns)
            self._units[unit.id] = unit
            cache_hit = False
            confidence = max(best_score, 0.0)

        self._reads_total += 1
        if cache_hit:
            self._reads_hit += 1

        evidence = list(unit.evidence)
        # Coverage = "does the unit actually answer THIS query?". Two-tier: cheap cosine over
        # per-claim embeddings decides the clear cases for free; the (more expensive)
        # coverage_scorer (cross-encoder / NLI / LLM entailment) is consulted ONLY in the
        # ambiguous band [coverage_floor, coverage_ceiling) where cosine can't tell "adjacent"
        # from "answers" — containment accuracy at a fraction of the per-query cost.
        coverage = self._semantic_coverage(qe, unit)
        if (
            self._coverage_scorer is not None
            and self._coverage_floor <= coverage < self._coverage_ceiling
        ):
            coverage = self._coverage_scorer(query, dict(unit.understanding))
        escalated = False
        # Semantic coverage gate: a HIT whose best per-claim match under-covers the query
        # escalates to fresh raw for THIS query (a retrieval, no LLM call) — still a hit.
        # Only when raw isn't already shipped AND escalation could surface it (so
        # CONTEXT_RAW/CONTEXT_ONLY don't waste a fetch), the switch is on, and the unit
        # isn't structured passthrough (already minimal — nothing to escalate to).
        if (
            cache_hit
            and self._enable_coverage_escalation
            and coverage < self._coverage_floor
            and not self._emits_raw(strat, escalated=False)
            and self._emits_raw(strat, escalated=True)
            and not unit.understanding.get("_passthrough")
        ):
            evidence = self._augment(evidence, self._retrieve(query, ns))
            escalated = True
            self._reads_escalated += 1

        return Result(
            understanding=dict(unit.understanding),
            evidence=evidence,
            cache_hit=cache_hit,
            unit_id=unit.id,
            confidence=confidence,
            namespace=ns,
            related=self._related(unit, qe, related, ns),
            context=self._project(unit.understanding, evidence, query, strat, escalated),
            coverage=coverage,
            escalated=escalated,
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

    def _best_match(self, qe: tuple[float, ...], ns: str) -> tuple[str | None, float]:
        best_id: str | None = None
        best = -1.0
        for unit_id, unit in self._units.items():
            if unit.namespace != ns:
                continue
            score = self._match_score(qe, unit)
            if score > best:
                best, best_id = score, unit_id
        return best_id, best

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
    ) -> None:
        chunks = self._retrieve(query, ns)
        synthesis = self._synth.synthesize(query, chunks)
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
        unit.touch()  # a build, not a hit -> no query recorded
        self._mark_fresh(unit)
        self._reindex(unit)
        self._persist(unit)

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
    ) -> dict[str, Any]:
        projected = SemanticCache._project_understanding(understanding, query)
        raw = (
            [chunk.text for chunk in evidence]
            if SemanticCache._emits_raw(strategy, escalated)
            else []
        )
        return {"understanding": projected, "raw": raw}

    @staticmethod
    def _project_understanding(understanding: dict[str, Any], query: str) -> dict[str, Any]:
        terms = set(tokenize(query))
        # Passthrough records (structured tool/API output) are already minimal — a
        # status/id/boolean with no lexical overlap is often THE decision-relevant
        # field, so never query-trim them; only LLM prose gets the relevance filter.
        passthrough = bool(understanding.get("_passthrough"))
        projected: dict[str, Any] = {}
        summary = understanding.get("summary")
        if summary is not None:
            projected["summary"] = summary
        claims = understanding.get("claims")
        if isinstance(claims, list):
            relevant = claims if passthrough else [c for c in claims if terms & set(tokenize(str(c)))]
            projected["claims"] = relevant or claims
        facts = understanding.get("facts")
        if isinstance(facts, dict):
            rel = facts if passthrough else {
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
    def stats(self) -> dict[str, float]:
        """Cache size + read-time observability. ``escalation_rate`` is the fraction of
        cache HITS that fell back to fresh raw — a high value means the cached
        understanding under-covers real queries (deepen it, or lower coverage_floor)."""
        hits = self._reads_hit
        return {
            "units": len(self._units),
            "tracked_artifacts": len(self._artifact_index),
            "reads": self._reads_total,
            "hits": hits,
            "escalations": self._reads_escalated,
            "escalation_rate": round(self._reads_escalated / hits, 3) if hits else 0.0,
            "hit_rate": round(hits / self._reads_total, 3) if self._reads_total else 0.0,
        }
