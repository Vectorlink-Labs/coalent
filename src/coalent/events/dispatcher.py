"""Event dispatcher: route a payload to its connector and feed the sink.

Transport-agnostic by design — it takes parsed payloads, so it is fully
testable with fixtures and can be driven by any HTTP framework (see
``coalent.events.http`` for an optional FastAPI receiver).
"""
from __future__ import annotations

from typing import Any, Callable, Iterable

from ..domain.models import ChangeEvent
from .connectors import EventConnector, default_connectors


class UnknownSourceError(KeyError):
    """Raised when a payload arrives for a source with no registered connector."""


class SignatureError(Exception):
    """Raised when signature verification fails for a signed payload."""


class EventDispatcher:
    """Parses payloads into change events and pushes them to a sink.

    ``sink`` is typically ``cache.invalidate`` but can be any callable — keeping
    the dispatcher decoupled from the cache implementation.
    """

    def __init__(
        self,
        sink: Callable[[ChangeEvent], Any],
        connectors: Iterable[EventConnector] | None = None,
    ) -> None:
        self._sink = sink
        chosen = list(connectors) if connectors is not None else default_connectors()
        self._connectors: dict[str, EventConnector] = {c.source: c for c in chosen}

    def register(self, connector: EventConnector) -> None:
        self._connectors[connector.source] = connector

    def dispatch(
        self,
        source: str,
        payload: dict[str, Any],
        *,
        raw_body: bytes | None = None,
        signature: str | None = None,
        secret: str | None = None,
    ) -> list[ChangeEvent]:
        """Verify (if a secret is given), parse, and emit events for one payload."""
        connector = self._connectors.get(source)
        if connector is None:
            raise UnknownSourceError(source)
        if secret is not None and not connector.verify(raw_body or b"", signature, secret):
            raise SignatureError(f"signature verification failed for source '{source}'")
        events = connector.parse(payload)
        for event in events:
            self._sink(event)
        return events
