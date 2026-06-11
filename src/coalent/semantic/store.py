"""Durable persistence for Cognition units (so the cache + its invalidation
graph survive a restart). ``CognitionStore`` is the port; in-memory and SQLite
implementations ship. The cache rebuilds its provenance/entity indexes from the
store on construction — fixing "invalidation dies after restart".
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from typing import Any, Protocol, runtime_checkable

from .serde import cognition_from_json, cognition_to_json
from .unit import Cognition

logger = logging.getLogger(__name__)

#: Exceptions a single corrupt/foreign stored record can raise on load — skipped
#: per-record so one poisoned key can't brick the cache's restart-time rebuild.
_DECODE_ERRORS = (json.JSONDecodeError, ValueError, KeyError, TypeError)


@runtime_checkable
class CognitionStore(Protocol):
    """Persistence for cognition units. Implementations must be thread-safe."""

    def get(self, unit_id: str) -> Cognition | None:
        ...

    def put(self, unit: Cognition) -> None:
        ...

    def delete(self, unit_id: str) -> None:
        ...

    def all(self) -> list[Cognition]:
        ...

    def __len__(self) -> int:
        ...


class InMemoryCognitionStore:
    """Ephemeral, in-process store (lost on exit)."""

    def __init__(self) -> None:
        self._units: dict[str, Cognition] = {}

    def get(self, unit_id: str) -> Cognition | None:
        return self._units.get(unit_id)

    def put(self, unit: Cognition) -> None:
        self._units[unit.id] = unit

    def delete(self, unit_id: str) -> None:
        self._units.pop(unit_id, None)

    def all(self) -> list[Cognition]:
        return list(self._units.values())

    def __len__(self) -> int:
        return len(self._units)


class SQLiteCognitionStore:
    """Durable SQLite-backed store — stdlib only. Pass a file path for
    persistence across restarts, or ``":memory:"`` for an ephemeral database."""

    def __init__(self, path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS cognition (id TEXT PRIMARY KEY, data TEXT NOT NULL)"
            )
            self._conn.commit()

    def get(self, unit_id: str) -> Cognition | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM cognition WHERE id = ?", (unit_id,)
            ).fetchone()
        return cognition_from_json(row[0]) if row else None

    def put(self, unit: Cognition) -> None:
        data = cognition_to_json(unit)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cognition (id, data) VALUES (?, ?)", (unit.id, data)
            )
            self._conn.commit()

    def delete(self, unit_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM cognition WHERE id = ?", (unit_id,))
            self._conn.commit()

    def all(self) -> list[Cognition]:
        with self._lock:
            rows = self._conn.execute("SELECT data FROM cognition").fetchall()
        units: list[Cognition] = []
        for row in rows:
            try:
                units.append(cognition_from_json(row[0]))
            except _DECODE_ERRORS:
                logger.warning("skipping unreadable cognition row in SQLite store")
        return units

    def __len__(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM cognition").fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class RedisCognitionStore:
    """Durable Redis-backed store — shared across processes and hosts.

    **Bring your own client** (a ``redis.Redis``) so we never pin a version: pass the
    client you already configured. Units are stored as JSON under ``prefix``. On
    construction the cache rebuilds its invalidation indexes from ``all()``, so
    freshness survives a restart — the same contract as :class:`SQLiteCognitionStore`.

        import redis
        store = RedisCognitionStore(redis.Redis.from_url("redis://localhost:6379/0"))
        # or the convenience constructor (needs `pip install coalent[redis]`):
        store = RedisCognitionStore.from_url("redis://localhost:6379/0")
    """

    def __init__(self, client: Any, *, prefix: str = "coalent:cognition:") -> None:
        self._redis = client
        self._prefix = prefix

    @classmethod
    def from_url(
        cls, url: str = "redis://localhost:6379/0", *, prefix: str = "coalent:cognition:"
    ) -> RedisCognitionStore:
        """Build a client from a URL (lazy ``redis`` import — extra ``coalent[redis]``)."""
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "RedisCognitionStore.from_url needs the 'redis' package: pip install coalent[redis]"
            ) from exc
        return cls(redis.Redis.from_url(url), prefix=prefix)

    def _key(self, unit_id: str) -> str:
        return f"{self._prefix}{unit_id}"

    def _match(self) -> str:
        """SCAN MATCH targeting the LITERAL prefix.

        SCAN's MATCH is a glob, so a prefix containing ``* ? [ ] \\`` (e.g. a tenant
        id like ``acme[prod]:``) would silently match the wrong keyset — and on
        restart rebuild an empty cache. Escape those metacharacters so the pattern
        matches the prefix verbatim.
        """
        escaped = "".join("\\" + ch if ch in "\\*?[]" else ch for ch in self._prefix)
        return f"{escaped}*"

    def _keys(self) -> set[str]:
        """Distinct keys under the prefix. SCAN may yield a key more than once
        (rehashing / concurrent writes), so de-duplicate — exact-count contract."""
        return set(self._redis.scan_iter(match=self._match()))

    @staticmethod
    def _text(data: Any) -> str:
        """Redis returns ``bytes`` unless the client decodes — normalize to ``str``."""
        return data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else str(data)

    def get(self, unit_id: str) -> Cognition | None:
        data = self._redis.get(self._key(unit_id))
        return cognition_from_json(self._text(data)) if data else None

    def put(self, unit: Cognition) -> None:
        self._redis.set(self._key(unit.id), cognition_to_json(unit))

    def delete(self, unit_id: str) -> None:
        self._redis.delete(self._key(unit_id))

    def all(self) -> list[Cognition]:
        units: list[Cognition] = []
        for key in self._keys():
            data = self._redis.get(key)
            if not data:
                continue
            try:
                units.append(cognition_from_json(self._text(data)))
            except _DECODE_ERRORS:
                # One poisoned/foreign key must not brick the restart-time rebuild.
                logger.warning("skipping unreadable cognition at redis key %r", key)
        return units

    def __len__(self) -> int:
        return len(self._keys())
