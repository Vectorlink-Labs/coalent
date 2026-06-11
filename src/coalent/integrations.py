"""Drop-in helpers for agent frameworks. Both are thin wrappers over ``get()`` —
plugging Coalent in is one call, so these exist for convenience, not necessity.

  * ``make_cognition_node`` — a LangGraph-compatible node (state -> partial state).
  * ``build_mcp_tools``     — transport-agnostic MCP tool specs over the read path.
"""
from __future__ import annotations

from typing import Any, Callable, TypedDict

from .semantic import SemanticCache


def make_cognition_node(
    cache: SemanticCache,
    *,
    query_key: str = "question",
    output_key: str = "context",
    namespace: str | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a LangGraph node bound to a cache.

    The node reads ``state[query_key]``, fetches fresh, decision-ready context, and
    writes it to ``state[output_key]``::

        graph.add_node("context", make_cognition_node(cache))
    """

    def node(state: dict[str, Any]) -> dict[str, Any]:
        result = cache.get(state.get(query_key, ""), namespace=namespace)
        return {output_key: result.context}

    return node


class ToolSpec(TypedDict):
    """A transport-agnostic description of one MCP tool."""

    name: str
    description: str
    handler: Callable[..., dict[str, Any]]
    input_schema: dict[str, Any]


def build_mcp_tools(cache: SemanticCache) -> list[ToolSpec]:
    """Return MCP tool specs wrapping the cache's read path, for any MCP runtime."""

    def get_context(query: str, namespace: str | None = None) -> dict[str, Any]:
        result = cache.get(query, namespace=namespace)
        return {
            "context": result.context,
            "raw": result.raw_text,
            "sources": [chunk.artifact_id for chunk in result.evidence],
            "cache_hit": result.cache_hit,
            "coverage": result.coverage,
        }

    return [
        ToolSpec(
            name="coalent.get_context",
            description="Fetch fresh, decision-ready context (understanding + sources) for a query.",
            handler=get_context,
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The user/agent question."},
                    "namespace": {"type": "string", "description": "Optional tenant/filter scope."},
                },
                "required": ["query"],
            },
        )
    ]
