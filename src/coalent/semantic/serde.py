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
from .unit import Cognition, QueryKey, ResidualSpan


def _chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    out: dict[str, Any] = {
        "artifact_id": chunk.artifact_id,
        "text": chunk.text,
        "version": chunk.version,
        "content_hash": chunk.content_hash,
    }
    # v0.7 ingest metadata — written ONLY when set, so meta-less chunks keep emitting
    # byte-identical pre-v0.7 JSON (and any older reader ignores the unknown key).
    if chunk.meta:
        out["meta"] = {str(k): str(v) for k, v in chunk.meta.items()}
    return out


def _chunk_from_dict(data: dict[str, Any]) -> Chunk:
    meta = data.get("meta")
    return Chunk(
        artifact_id=data["artifact_id"],
        text=data["text"],
        version=data.get("version", ""),
        content_hash=data.get("content_hash", ""),
        meta={str(k): str(v) for k, v in meta.items()} if isinstance(meta, dict) else None,
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


def _residual_span_to_dict(span: ResidualSpan) -> dict[str, Any]:
    # Deliberately NO embedding: it is regenerable (~30KB of JSON per span otherwise) —
    # the cache re-embeds loaded spans in one batch on first side-channel use.
    return {
        "text": span.text,
        "artifact_id": span.artifact_id,
        "chunk_idx": span.chunk_idx,
    }


def _residual_span_from_dict(data: dict[str, Any]) -> ResidualSpan:
    return ResidualSpan(
        text=data["text"],
        artifact_id=data.get("artifact_id", ""),
        chunk_idx=int(data.get("chunk_idx", -1)),
        embedding=tuple(float(x) for x in data.get("embedding", [])),
    )


def cognition_to_dict(unit: Cognition) -> dict[str, Any]:
    payload = {
        "id": unit.id,
        "namespace": unit.namespace,
        "query": unit.query,
        "query_embedding": list(unit.query_embedding),
        "understanding": unit.understanding,
        "evidence": [_chunk_to_dict(c) for c in unit.evidence],
        "provenance": _manifest_to_dict(unit.provenance),
        # v0.3 — keyed by understanding; per-claim coverage; behavioral hit log.
        "understanding_embedding": list(unit.understanding_embedding),
        "claim_embeddings": [list(c) for c in unit.claim_embeddings],
        "synth_tokens": unit.synth_tokens,
        "hit_queries": list(unit.hit_queries),
        "status": unit.status.value,
        "freshness_epoch": unit.freshness_epoch,
        "hits": unit.hits,
        "created_at": unit.created_at,
        "last_access": unit.last_access,
    }
    # v0.6 tier-2 residual spans — written ONLY when set, so a cache that never opted in
    # keeps emitting byte-identical v0.5 JSON (and any v0.5 reader ignores unknown keys).
    if unit.residual_spans:
        payload["residual_spans"] = [_residual_span_to_dict(s) for s in unit.residual_spans]
    if unit.span_hits:
        payload["span_hits"] = unit.span_hits
    if unit.lossy:
        payload["lossy"] = True
    # v0.7 ingest metadata — same written-only-when-set contract as the v0.6 keys above:
    # a store that never saw Chunk.meta keeps emitting byte-identical v0.6 JSON.
    if unit.source_meta:
        payload["source_meta"] = dict(unit.source_meta)
    # v0.6 query keys: CONFIRMED keys only (provisional ones expire with the in-process read
    # ring — never durable). Keys carry no text, so the embedding IS the key and must be
    # persisted (float-tuple style, capped at 8/unit by the cache — the size tradeoff).
    confirmed = [k for k in unit.query_keys if not k.read_id]
    if confirmed:
        payload["query_keys"] = [
            {"embedding": list(k.embedding), "claim_idx": k.claim_idx,
             "span_text": k.span_text, "hits": k.hits}
            for k in confirmed
        ]
    return payload


def cognition_from_dict(data: dict[str, Any]) -> Cognition:
    return Cognition(
        id=data["id"],
        namespace=data.get("namespace", ""),
        query=data.get("query", ""),
        query_embedding=tuple(float(x) for x in data.get("query_embedding", [])),
        understanding=data.get("understanding", {}),
        evidence=tuple(_chunk_from_dict(c) for c in data.get("evidence", [])),
        provenance=_manifest_from_dict(data["provenance"]),
        # v0.3 — absent in v0.2 JSON: default to empty (-> needs_backfill on load).
        understanding_embedding=tuple(float(x) for x in data.get("understanding_embedding", [])),
        claim_embeddings=tuple(
            tuple(float(x) for x in c) for c in data.get("claim_embeddings", [])
        ),
        synth_tokens=int(data.get("synth_tokens", 0)),  # absent in pre-v0.4 JSON -> 0
        hit_queries=tuple(str(q) for q in data.get("hit_queries", [])),
        # v0.6 — absent in pre-v0.6 JSON: no spans, no lossy state (old dicts load fine).
        residual_spans=tuple(
            _residual_span_from_dict(s) for s in data.get("residual_spans", [])
        ),
        span_hits=int(data.get("span_hits", 0)),
        lossy=bool(data.get("lossy", False)),
        # v0.7 — absent in pre-v0.7 JSON: no ingest metadata (old dicts load fine).
        source_meta={str(k): str(v) for k, v in (data.get("source_meta") or {}).items()},
        query_keys=tuple(
            QueryKey(
                embedding=tuple(float(x) for x in k.get("embedding", [])),
                claim_idx=int(k.get("claim_idx", -1)),
                span_text=str(k.get("span_text", "")),
                hits=int(k.get("hits", 0)),
            )
            for k in data.get("query_keys", [])
        ),
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
