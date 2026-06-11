"""Tests for the LangGraph node + MCP tool helpers."""
from __future__ import annotations

from coalent import build_mcp_tools, make_cognition_node
from coalent.semantic import InMemoryRetriever, SemanticCache, StubSynthesizer


def _cache() -> SemanticCache:
    retriever = InMemoryRetriever()
    retriever.add("confluence:hr", "Leave policy: 21 days of annual leave per year.")
    return SemanticCache(retriever, StubSynthesizer())


def test_langgraph_node() -> None:
    node = make_cognition_node(_cache())
    out = node({"question": "what is our leave policy?"})
    assert "context" in out
    assert "understanding" in out["context"]


def test_langgraph_node_custom_keys() -> None:
    node = make_cognition_node(_cache(), query_key="q", output_key="ctx")
    out = node({"q": "leave policy"})
    assert "ctx" in out


def test_mcp_tool() -> None:
    tools = build_mcp_tools(_cache())
    spec = tools[0]
    assert spec["name"] == "coalent.get_context"
    assert "query" in spec["input_schema"]["properties"]

    result = spec["handler"]("what is our leave policy?")
    assert "context" in result
    assert "confluence:hr" in result["sources"]
    assert "21 days" in result["raw"]
