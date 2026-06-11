"""Event layer.

Turns native source payloads (GitHub, deploy, Jira, generic CDC) into normalized
:class:`~coalent.domain.models.ChangeEvent`s and feeds them to a sink
(typically ``cache.invalidate``). This is what makes "always fresh" automatic:
a merged PR or a deploy dirties exactly the cognition derived from it.
"""
from __future__ import annotations

from .connectors import (
    DeployConnector,
    EventConnector,
    GenericCDCConnector,
    GitHubConnector,
    JiraConnector,
    default_connectors,
)
from .dispatcher import EventDispatcher, SignatureError, UnknownSourceError
from .signatures import compute_github_signature, verify_github_signature

__all__ = [
    "EventConnector",
    "GitHubConnector",
    "DeployConnector",
    "JiraConnector",
    "GenericCDCConnector",
    "default_connectors",
    "EventDispatcher",
    "UnknownSourceError",
    "SignatureError",
    "compute_github_signature",
    "verify_github_signature",
]
