"""Core domain primitives — pure data, no external dependencies.

  - SourceSpan / ProvenanceManifest : the source lineage a cognition unit records.
  - ChangeEvent                     : a normalized "artifact A changed" signal.
  - Status                          : a unit's lifecycle state.

These are framework-agnostic so the cache plugs into any runtime.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum


class Status(str, Enum):
    """Lifecycle state of a cognition unit."""

    EMPTY = "empty"
    MATERIALIZING = "materializing"
    FRESH = "fresh"
    DIRTY = "dirty"


def _hash_text(text: str) -> str:
    """Stable short content hash used to detect whether a span changed."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """A precise, addressable pointer to a piece of source.

    ``artifact_id`` is what change events reference; the ``version`` / ``span`` /
    ``content_hash`` triple lets us detect whether the *specific* content a unit
    was derived from has actually changed.
    """

    artifact_id: str            # e.g. "confluence:98231"
    version: str = ""           # revision / commit sha / deploy id
    span: str | None = None     # e.g. "L40-L88" (None = whole artifact)
    content_hash: str = ""      # hash of the exact span content

    def identity(self) -> str:
        """A fully-qualified identity string for change comparison."""
        return f"{self.artifact_id}@{self.version}#{self.span or '*'}:{self.content_hash}"

    def key(self) -> str:
        """Stable, version-independent key for this span (artifact + range)."""
        return f"{self.artifact_id}#{self.span or '*'}"

    @classmethod
    def from_text(
        cls,
        artifact_id: str,
        text: str,
        *,
        version: str = "",
        span: str | None = None,
    ) -> SourceSpan:
        """Build a span and compute its content hash from raw text."""
        return cls(artifact_id=artifact_id, version=version, span=span,
                   content_hash=_hash_text(text))


@dataclass(frozen=True, slots=True)
class ProvenanceManifest:
    """The exact derivation inputs of a cognition unit — its source lineage.

    Simultaneously the freshness key and the invalidation graph: invalidation is
    pure set-membership over ``artifact_ids()``.
    """

    model_version: str
    prompt_version: str
    source_spans: tuple[SourceSpan, ...] = ()
    # Artifacts read but not central to the derivation (blast-radius tracking).
    observed_edges: tuple[str, ...] = ()

    def artifact_ids(self) -> frozenset[str]:
        """Every artifact this unit depends on (spans + observed edges)."""
        ids = {span.artifact_id for span in self.source_spans}
        ids.update(self.observed_edges)
        return frozenset(ids)

    def span_for(self, artifact_id: str) -> SourceSpan | None:
        """Return the first tracked span for an artifact (or ``None``)."""
        for span in self.source_spans:
            if span.artifact_id == artifact_id:
                return span
        return None

    def spans_for(self, artifact_id: str) -> tuple[SourceSpan, ...]:
        """Return *every* tracked span for an artifact (span-level lookup)."""
        return tuple(s for s in self.source_spans if s.artifact_id == artifact_id)

    def fingerprint(self) -> str:
        """A deterministic fingerprint of the whole derivation context."""
        hasher = hashlib.sha256()
        hasher.update(self.model_version.encode("utf-8"))
        hasher.update(self.prompt_version.encode("utf-8"))
        for span in sorted(self.source_spans, key=lambda s: s.artifact_id):
            hasher.update(span.identity().encode("utf-8"))
        return hasher.hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class ChangeEvent:
    """A normalized "artifact A changed" event.

    Connectors for GitHub / Jira / deploys / CDC translate their native payloads
    into this single shape, which is all invalidation needs.
    """

    artifact_id: str
    version: str = ""
    span: str | None = None
    content_hash: str = ""
    kind: str = "update"             # e.g. "github.push", "deploy", "delete"
    ts: float = field(default_factory=time.time)
