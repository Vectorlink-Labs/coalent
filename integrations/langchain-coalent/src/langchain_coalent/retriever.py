"""``CoalentRetriever`` — the cache as a LangChain retriever.

Drop-in for any chain/agent that takes a retriever: ``invoke(query)`` runs
``cache.get()`` and returns the served payload as ``Document``(s) whose metadata
carries ``{read_id, sources, cache_hit}`` — everything the refusal loop
(``cache.report_refusal(read_id)`` / ``report_success(read_id)``) needs.
"""
from __future__ import annotations

import json
from typing import Any

from coalent import Result, SemanticCache
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict


def render_payload(result: Result) -> str:
    """Render a ``Result`` as the one context string an answerer should read.

    Mirrors the shipped MCP server's rendering: the pool payload (attributed,
    budget-packed fact lines) when the pool read path served one; otherwise the
    unit path's projection dict as compact JSON (never invented prose); plus the
    attributed raw lines whenever the RAG floor fired.
    """
    parts: list[str] = []
    pool_text = str(result.context.get("pool", "") or "")
    if pool_text:
        parts.append(pool_text)
    else:
        unit_ctx = {k: v for k, v in result.context.items() if k != "raw"}
        if unit_ctx:
            parts.append(json.dumps(unit_ctx, ensure_ascii=False))
    raw = result.context.get("raw")
    if isinstance(raw, list) and raw:
        parts.append("\n\n".join(str(r) for r in raw))
    return "\n\n".join(parts)


def sources_of(cache: SemanticCache, result: Result) -> list[str]:
    """The artifact ids behind a read, served order first, de-duplicated.

    Each served pool claim's owning unit contributes its evidence artifacts (via
    the public ``drill``), then any escalation raw evidence.
    """
    out: list[str] = []
    for claim in result.pool:
        for chunk in cache.drill(claim.unit_id):
            if chunk.artifact_id and chunk.artifact_id not in out:
                out.append(chunk.artifact_id)
    for chunk in result.evidence:
        if chunk.artifact_id and chunk.artifact_id not in out:
            out.append(chunk.artifact_id)
    return out


class CoalentRetriever(BaseRetriever):
    """A LangChain ``BaseRetriever`` over a Coalent :class:`~coalent.SemanticCache`.

    ``invoke(query)`` returns one primary ``Document`` whose ``page_content`` is
    the served payload (see :func:`render_payload`) and whose metadata carries:

    * ``read_id`` — hand it to ``cache.report_refusal()`` when your answerer
      refuses over this payload (and ``report_success()`` after a good retry);
    * ``sources`` — the artifact ids behind the read (provenance);
    * ``cache_hit`` — whether the read served from cache (no synthesis call);
    * plus ``coverage``, ``escalated``, ``unit_id``, ``namespace`` for observability.

    With ``include_evidence=True`` the retained raw evidence follows as one
    ``Document`` per chunk (``metadata: {source, version, coalent_evidence: True}``).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cache: SemanticCache
    """The Coalent cache to read through (see ``create_coalent_cache``)."""
    namespace: str | None = None
    """Optional Coalent namespace forwarded to every ``cache.get()``."""
    related: int = 3
    """How many related units ``cache.get()`` may fold in."""
    include_evidence: bool = False
    """Also return the retained raw evidence chunks as extra Documents."""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        result = self.cache.get(query, namespace=self.namespace, related=self.related)
        metadata: dict[str, Any] = {
            "read_id": result.read_id,
            "sources": sources_of(self.cache, result),
            "cache_hit": result.cache_hit,
            "coverage": result.coverage,
            "escalated": result.escalated,
            "unit_id": result.unit_id,
            "namespace": result.namespace,
            # v0.7 read surface (additive keys; the boundary port): the read's own
            # doubt, so a chain-side evaluator can act without drilling. gaps is
            # non-empty only when the cache was built with gap_detector=True.
            "needs_retrieval": result.needs_retrieval,
            "gaps": list(result.gaps),
        }
        docs = [Document(page_content=render_payload(result), metadata=metadata)]
        if self.include_evidence:
            docs.extend(
                Document(
                    page_content=chunk.text,
                    metadata={
                        "source": chunk.artifact_id,
                        "version": chunk.version,
                        "coalent_evidence": True,
                    },
                )
                for chunk in result.evidence
            )
        return docs
