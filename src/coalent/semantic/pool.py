"""The claim pool index (v0.6 M1) — the storage port the pool-first read path scans.

A :class:`ClaimIndex` holds one row per atomic claim of every cached unit in ONE
namespace, keyed by the claim's embedding, so a read can rank the whole cache's
claims against a query instead of matching a single unit. The built-in
:class:`LocalClaimIndex` runs numpy when it is importable and a pure-Python twin
otherwise (the zero-dependency covenant); both apply the SAME contract tie-break
``(-score, unit_id, claim_idx)`` and evaluate freshness as a LIVE pull-mask — a
unit's rows vanish from results the instant its owner goes stale, never
snapshotted into the index at add time.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

try:  # optional acceleration (``pip install coalent[fast]``); pure path is the twin.
    import numpy as _np
except ImportError:  # pragma: no cover - environment-dependent
    _np = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class ClaimRef:
    """A reference to one indexed claim row: its owner unit + position + text."""

    unit_id: str
    claim_idx: int      # position in the owner's claim list AT ADD TIME (regenerated on rebuild)
    text: str


@runtime_checkable
class ClaimIndex(Protocol):
    """The claim-pool storage port (one instance per namespace).

    ``add`` has REPLACE semantics — re-adding a unit id swaps that unit's rows
    atomically (a rebuild). ``search`` returns at most ``top_n`` rows sorted by the
    contract tie-break ``(-score, unit_id, claim_idx)`` and, with the default
    ``include_stale=False``, never returns a masked/stale row. Freshness is a live
    pull-mask, so ``mask``/``unmask`` are advisory (no-ops for an in-process index
    whose authority is the live unit status; load-bearing only for persistent
    adapters). Implementations must never raise on an unknown unit id in
    ``remove``/``mask``/``unmask``.
    """

    def add(self, unit_id: str, claims: Sequence[str],
            embs: Sequence[Sequence[float]]) -> int: ...       # REPLACE; returns rows indexed

    def remove(self, unit_id: str) -> None: ...

    def mask(self, unit_id: str) -> None: ...                  # no-op for unknown ids, never raises

    def unmask(self, unit_id: str) -> None: ...

    def search(self, qe: Sequence[float], top_n: int,
               *, include_stale: bool = False
               ) -> list[tuple[float, ClaimRef, bool]]: ...    # (score, ref, fresh); cosine in [-1,1]

    def __len__(self) -> int: ...                              # indexed row count


def _l2_normalize(vec: Sequence[float]) -> tuple[float, ...]:
    """Unit-length copy of ``vec`` (a zero vector is returned unchanged)."""
    fv = tuple(float(v) for v in vec)
    norm = math.sqrt(sum(v * v for v in fv))
    if norm == 0.0:
        return fv
    return tuple(v / norm for v in fv)


class LocalClaimIndex:
    """Built-in in-process claim index — numpy when available, pure-Python otherwise.

    Rows are stored per owner (add = replace that owner's rows); a flattened view is
    rebuilt lazily, invalidated by an internal monotone epoch that bumps on every
    add/remove (never a length/hash marker, so a same-size rebuild can never restore
    a prior marker). Freshness is applied at SEARCH time from the live ``fresh_of``
    callback, evaluated once per distinct owner per scan — never cached into a row —
    so a stale owner's claims are invisible the instant its status flips, with no
    re-add. Both compute paths funnel scored rows through the identical contract
    tie-break ``(-score, unit_id, claim_idx)``, so the numpy and pure paths return
    bit-for-bit the same ClaimRef sequence (scores agree to the float32 bound).
    """

    def __init__(
        self,
        *,
        fresh_of: Callable[[str], bool] | None = None,
        use_numpy: bool | None = None,
    ) -> None:
        # ``fresh_of(unit_id) -> bool`` is the live freshness authority; None = all fresh.
        self._fresh_of = fresh_of
        if use_numpy is None:
            self._use_numpy = _np is not None
        else:
            self._use_numpy = bool(use_numpy) and _np is not None
        # Authoritative per-owner rows: unit_id -> list of (claim_idx, text, normalized_emb).
        self._rows_by_unit: dict[str, list[tuple[int, str, tuple[float, ...]]]] = {}
        self._dim: int | None = None
        self._epoch = 0                 # monotone; bumps on any row mutation
        self._built_epoch = -1          # epoch the flattened view was last built at
        self._flat_unit_ids: list[str] = []
        self._flat_claim_idxs: list[int] = []
        self._flat_texts: list[str] = []
        self._flat_embs: list[tuple[float, ...]] = []
        self._matrix: Any = None        # numpy float32 matrix (built lazily) or None

    # ------------------------------------------------------------------ mutation
    def add(self, unit_id: str, claims: Sequence[str],
            embs: Sequence[Sequence[float]]) -> int:
        """Replace ``unit_id``'s rows with ``claims``/``embs`` (parallel). Skips blank
        texts and empty/all-zero embeddings; a claim/emb length mismatch or a dimension
        change after the first add raises ``ValueError``. Returns the rows indexed."""
        if len(claims) != len(embs):
            raise ValueError(
                f"claims/embs length mismatch: {len(claims)} != {len(embs)}"
            )
        rows: list[tuple[int, str, tuple[float, ...]]] = []
        for idx, (text, emb) in enumerate(zip(claims, embs)):
            t = str(text).strip()
            fv = tuple(float(v) for v in emb)
            if not t or not fv or all(v == 0.0 for v in fv):
                continue                # blank text / empty / zero vector -> not servable
            if self._dim is None:
                self._dim = len(fv)
            elif len(fv) != self._dim:
                raise ValueError(
                    f"embedding dim {len(fv)} != index dim {self._dim}; the embedder "
                    "changed — create a fresh index (or a new claim_index factory) after "
                    "an embedder swap"
                )
            rows.append((idx, t, _l2_normalize(fv)))
        if rows:
            self._rows_by_unit[unit_id] = rows        # REPLACE
        else:
            self._rows_by_unit.pop(unit_id, None)     # nothing servable -> drop the owner
        self._epoch += 1
        return len(rows)

    def remove(self, unit_id: str) -> None:
        """Drop ``unit_id``'s rows entirely (a delete). No-op for an unknown id."""
        if self._rows_by_unit.pop(unit_id, None) is not None:
            self._epoch += 1

    def mask(self, unit_id: str) -> None:
        """Advisory: in-process freshness authority is the live ``fresh_of`` callback,
        so masking is a no-op here (load-bearing only for persistent adapters)."""
        return None

    def unmask(self, unit_id: str) -> None:
        """Advisory counterpart to :meth:`mask` — a no-op in process."""
        return None

    def __len__(self) -> int:
        return sum(len(rows) for rows in self._rows_by_unit.values())

    # --------------------------------------------------------------------- search
    def _rebuild(self) -> None:
        """Flatten per-owner rows into parallel arrays (+ numpy matrix) — lazily,
        only when the epoch advanced past the last build."""
        if self._built_epoch == self._epoch:
            return
        uids: list[str] = []
        cidx: list[int] = []
        texts: list[str] = []
        embs: list[tuple[float, ...]] = []
        for uid, rows in self._rows_by_unit.items():
            for claim_idx, text, emb in rows:
                uids.append(uid)
                cidx.append(claim_idx)
                texts.append(text)
                embs.append(emb)
        self._flat_unit_ids = uids
        self._flat_claim_idxs = cidx
        self._flat_texts = texts
        self._flat_embs = embs
        if self._use_numpy and _np is not None and embs:
            self._matrix = _np.asarray(embs, dtype=_np.float32)
        else:
            self._matrix = None
        self._built_epoch = self._epoch

    def search(self, qe: Sequence[float], top_n: int,
               *, include_stale: bool = False
               ) -> list[tuple[float, ClaimRef, bool]]:
        """Top ``top_n`` claim rows by cosine to ``qe``, freshest-first-masked.

        Freshness is pulled live from ``fresh_of`` per distinct owner (cached only for
        the duration of THIS scan). ``include_stale=True`` also returns masked rows,
        each flagged ``fresh=False`` — telemetry only. Both compute paths apply the
        identical contract tie-break, so results are order-equivalent."""
        self._rebuild()
        if top_n <= 0 or not self._flat_embs:
            return []
        if self._use_numpy and self._matrix is not None:
            q = _np.asarray(qe, dtype=_np.float32)
            norm = float(_np.linalg.norm(q))
            q = q / (norm if norm else 1.0)
            scores: list[float] = [float(s) for s in (self._matrix @ q).tolist()]
        else:
            qn = _l2_normalize(qe)
            scores = [
                sum(a * b for a, b in zip(qn, emb)) for emb in self._flat_embs
            ]
        fresh_cache: dict[str, bool] = {}

        def _is_fresh(uid: str) -> bool:
            if self._fresh_of is None:
                return True
            cached = fresh_cache.get(uid)
            if cached is None:
                cached = bool(self._fresh_of(uid))
                fresh_cache[uid] = cached
            return cached

        # Score every row, apply the live freshness pull-mask, then the contract tie-break.
        scored: list[tuple[float, str, int, str, bool]] = []
        for i, score in enumerate(scores):
            uid = self._flat_unit_ids[i]
            fresh = _is_fresh(uid)
            if not include_stale and not fresh:
                continue
            scored.append((score, uid, self._flat_claim_idxs[i], self._flat_texts[i], fresh))
        scored.sort(key=lambda r: (-r[0], r[1], r[2]))   # (-score, unit_id, claim_idx)
        return [
            (score, ClaimRef(uid, claim_idx, text), fresh)
            for score, uid, claim_idx, text, fresh in scored[:top_n]
        ]
