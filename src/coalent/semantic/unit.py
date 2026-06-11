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
    status: Status = Status.FRESH
    freshness_epoch: float = field(default_factory=time.time)
    hits: int = 0
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)

    @property
    def is_fresh(self) -> bool:
        return self.status == Status.FRESH

    def touch(self) -> None:
        self.hits += 1
        self.last_access = time.time()

    def mark_dirty(self) -> None:
        self.status = Status.DIRTY

    def mark_fresh(self) -> None:
        self.status = Status.FRESH
        self.freshness_epoch = time.time()
