"""Source connectors: native payload -> normalized ChangeEvents.

Each connector knows one source format and is tolerant of partial payloads
(missing keys produce no events rather than raising). Only GitHub implements
signature verification; the rest accept all (verify at the transport layer or
via a shared secret as needed).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from ..domain.models import ChangeEvent
from .signatures import verify_github_signature


class EventConnector(ABC):
    """Base class for a source-specific webhook/CDC parser."""

    source: ClassVar[str]

    @abstractmethod
    def parse(self, payload: dict[str, Any]) -> list[ChangeEvent]:
        """Translate a native payload into normalized change events."""

    def verify(self, raw_body: bytes, signature: str | None, secret: str) -> bool:
        """Verify a signed payload. Default: no signature scheme -> accept."""
        return True


class GitHubConnector(EventConnector):
    """Parses GitHub ``push`` (and PR payloads that carry a file list)."""

    source = "github"

    def parse(self, payload: dict[str, Any]) -> list[ChangeEvent]:
        pull_request = payload.get("pull_request") or {}
        version = str(payload.get("after") or pull_request.get("merge_commit_sha") or "")

        paths: set[str] = set()
        for commit in payload.get("commits") or []:
            for kind in ("added", "modified", "removed"):
                for path in commit.get(kind) or []:
                    paths.add(str(path))
        for entry in pull_request.get("files") or []:
            paths.add(str(entry.get("filename", "")) if isinstance(entry, dict) else str(entry))
        paths.discard("")

        return [
            ChangeEvent(artifact_id=f"github:{path}", version=version, kind="github.push")
            for path in sorted(paths)
        ]

    def verify(self, raw_body: bytes, signature: str | None, secret: str) -> bool:
        return verify_github_signature(secret, raw_body, signature)


class DeployConnector(EventConnector):
    """Parses a deploy event: ``{"service": ..., "version": ...}``."""

    source = "deploy"

    def parse(self, payload: dict[str, Any]) -> list[ChangeEvent]:
        service = payload.get("service")
        if not service:
            return []
        return [
            ChangeEvent(
                artifact_id=f"deploy:{service}",
                version=str(payload.get("version", "")),
                kind="deploy",
            )
        ]


class JiraConnector(EventConnector):
    """Parses a Jira issue webhook: ``{"issue": {"key": ...}}``."""

    source = "jira"

    def parse(self, payload: dict[str, Any]) -> list[ChangeEvent]:
        issue = payload.get("issue") or {}
        key = issue.get("key")
        if not key:
            return []
        fields = issue.get("fields") or {}
        return [
            ChangeEvent(
                artifact_id=f"jira:{key}",
                version=str(fields.get("updated", "")),
                kind="jira.update",
            )
        ]


class GenericCDCConnector(EventConnector):
    """Generic change-data-capture: a single change dict or a ``changes`` list.

    Each item: ``{artifact_id, version?, span?, content_hash?, kind?}``.
    """

    source = "cdc"

    def parse(self, payload: dict[str, Any]) -> list[ChangeEvent]:
        raw = payload.get("changes")
        items = raw if isinstance(raw, list) else [payload]
        events: list[ChangeEvent] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            artifact_id = item.get("artifact_id")
            if not artifact_id:
                continue
            events.append(
                ChangeEvent(
                    artifact_id=str(artifact_id),
                    version=str(item.get("version", "")),
                    span=item.get("span"),
                    content_hash=str(item.get("content_hash", "")),
                    kind=str(item.get("kind", "cdc")),
                )
            )
        return events


def default_connectors() -> list[EventConnector]:
    """The built-in connector set."""
    return [GitHubConnector(), DeployConnector(), JiraConnector(), GenericCDCConnector()]
