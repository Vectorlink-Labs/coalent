"""Tests for JSONPassthroughSynthesizer — cache structured tool output as-is.

Proves:
  1. a JSON object becomes the unit's ``facts`` with NO LLM call.
  2. all chunks are cited, so provenance + freshness still work (a changed tool
     result invalidates the unit exactly like a doc edit).
  3. non-JSON text degrades to a raw claim (never dropped).
  4. multiple records merge (later wins) or stay separate with ``merge=False``.
"""
from __future__ import annotations

import json

from coalent.semantic import (
    Chunk,
    FunctionRetriever,
    JSONPassthroughSynthesizer,
    SemanticCache,
)


def test_json_object_becomes_facts_without_an_llm() -> None:
    synth = JSONPassthroughSynthesizer()
    chunks = [Chunk("tool:leave_balance", json.dumps({"annual_leave": 12, "sick": 5}))]

    result = synth.synthesize("leave balance?", chunks)

    assert result.ok is True
    assert result.understanding["facts"]["annual_leave"] == 12
    assert result.used == [0]  # cited -> provenance + freshness work


def test_non_json_falls_back_to_a_claim() -> None:
    synth = JSONPassthroughSynthesizer()
    result = synth.synthesize("x", [Chunk("tool:note", "just a plain string")])
    assert "just a plain string" in result.understanding["claims"]


def test_records_merge_or_stay_separate() -> None:
    chunks = [
        Chunk("tool:a", json.dumps({"k": 1})),
        Chunk("tool:b", json.dumps({"k": 2})),
    ]
    merged = JSONPassthroughSynthesizer().synthesize("x", chunks)
    assert merged.understanding["facts"]["k"] == 2  # later wins

    separate = JSONPassthroughSynthesizer(merge=False).synthesize("x", chunks)
    assert separate.understanding["facts"] == {}
    assert {"k": 1} in separate.understanding["claims"]


def test_cached_tool_unit_is_invalidated_by_source_change() -> None:
    balance = '{"employee": "A", "annual_leave": 12}'
    retriever = FunctionRetriever(lambda q, ns: [Chunk("tool:balance:A", balance)])
    cache = SemanticCache(retriever, JSONPassthroughSynthesizer())

    result = cache.get("leave balance for A")
    assert result.understanding["facts"]["annual_leave"] == 12

    # the tool result changed (revalidation / webhook) -> unit dirties, rebuilds next read
    changed = cache.source_changed("tool:balance:A", text='{"employee": "A", "annual_leave": 8}')
    assert result.unit_id in changed.dirtied


def test_passthrough_fields_survive_projection_on_partial_query_match() -> None:
    """Structured fields with NO query overlap (status/ids/flags) must NOT be trimmed
    from the projected context — they're often the decision-relevant ones."""
    record = json.dumps(
        {"employee": "alice", "annual_leave_remaining": 5, "sick_remaining": 3, "status": "active"}
    )
    retriever = FunctionRetriever(lambda q, ns: [Chunk("tool:balance:alice", record)])
    cache = SemanticCache(retriever, JSONPassthroughSynthesizer())

    # this query lexically matches leave/remaining/alice but NOT "status"
    result = cache.get("how many leave days remaining for alice")
    facts = result.context["understanding"]["facts"]   # the payload the integrations surface
    assert facts["status"] == "active"                 # survives despite no query overlap
    assert facts["annual_leave_remaining"] == 5


def test_literal_json_null_is_kept_as_value_not_raw_string() -> None:
    result = JSONPassthroughSynthesizer().synthesize("x", [Chunk("tool:none", "null")])
    assert None in result.understanding["claims"]      # structured null, not the string
    assert "null" not in result.understanding["claims"]
