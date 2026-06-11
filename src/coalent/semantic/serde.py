"""(De)serialization of Cognition units to/from plain JSON for persistence.

Keeps the full unit — understanding, retained evidence, query embedding, and
provenance — so a restart restores both the cache and its invalidation graph.
``understanding`` is assumed JSON-serializable (the built-in synthesizers emit
JSON-safe values).
"""
from __future__ import annotations

import json
from typing import Any

from ..domain.models import ProvenanceManifest, SourceSpan, Status
from .ports import Chunk
from .unit import Cognition


def _chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    return {
        "artifact_id": chunk.artifact_id,
        "text": chunk.text,
        "version": chunk.version,
        "content_hash": chunk.content_hash,
    }


def _chunk_from_dict(data: dict[str, Any]) -> Chunk:
    return Chunk(
        artifact_id=data["artifact_id"],
        text=data["text"],
        version=data.get("version", ""),
        content_hash=data.get("content_hash", ""),
    )


def _span_to_dict(span: SourceSpan) -> dict[str, Any]:
    return {
        "artifact_id": span.artifact_id,
        "version": span.version,
        "span": span.span,
        "content_hash": span.content_hash,
    }


def _span_from_dict(data: dict[str, Any]) -> SourceSpan:
    return SourceSpan(
        artifact_id=data["artifact_id"],
        version=data.get("version", ""),
        span=data.get("span"),
        content_hash=data.get("content_hash", ""),
    )


def _manifest_to_dict(manifest: ProvenanceManifest) -> dict[str, Any]:
    return {
        "model_version": manifest.model_version,
        "prompt_version": manifest.prompt_version,
        "source_spans": [_span_to_dict(s) for s in manifest.source_spans],
        "observed_edges": list(manifest.observed_edges),
    }


def _manifest_from_dict(data: dict[str, Any]) -> ProvenanceManifest:
    return ProvenanceManifest(
        model_version=data["model_version"],
        prompt_version=data["prompt_version"],
        source_spans=tuple(_span_from_dict(s) for s in data.get("source_spans", [])),
        observed_edges=tuple(data.get("observed_edges", [])),
    )


def cognition_to_dict(unit: Cognition) -> dict[str, Any]:
    return {
        "id": unit.id,
        "namespace": unit.namespace,
        "query": unit.query,
        "query_embedding": list(unit.query_embedding),
        "understanding": unit.understanding,
        "evidence": [_chunk_to_dict(c) for c in unit.evidence],
        "provenance": _manifest_to_dict(unit.provenance),
        "status": unit.status.value,
        "freshness_epoch": unit.freshness_epoch,
        "hits": unit.hits,
        "created_at": unit.created_at,
        "last_access": unit.last_access,
    }


def cognition_from_dict(data: dict[str, Any]) -> Cognition:
    return Cognition(
        id=data["id"],
        namespace=data.get("namespace", ""),
        query=data.get("query", ""),
        query_embedding=tuple(float(x) for x in data.get("query_embedding", [])),
        understanding=data.get("understanding", {}),
        evidence=tuple(_chunk_from_dict(c) for c in data.get("evidence", [])),
        provenance=_manifest_from_dict(data["provenance"]),
        status=Status(data.get("status", "fresh")),
        freshness_epoch=data.get("freshness_epoch", 0.0),
        hits=data.get("hits", 0),
        created_at=data.get("created_at", 0.0),
        last_access=data.get("last_access", 0.0),
    )


def cognition_to_json(unit: Cognition) -> str:
    return json.dumps(cognition_to_dict(unit))


def cognition_from_json(payload: str) -> Cognition:
    return cognition_from_dict(json.loads(payload))
