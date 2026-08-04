"""LG2 — e2e chain: cold build then warm hit through CoalentRetriever.
LG3 — freshness: source_changed on the wrapped store's doc id -> next retrieval rebuilds.

Fully offline: fake VectorStore + deterministic fake embeddings + scripted fake chat
model. The scripted extractor echoes any known corpus sentence it can see in the
synthesis prompt, so payload content exactly tracks what was retrieved and rebuilt.
"""
from __future__ import annotations

import json
import re
from typing import Any

import pytest
from conftest import BagOfWordsEmbeddings, FakeVectorStore, ScriptedChatModel

from coalent import SemanticCache
from langchain_coalent import CoalentRetriever, create_coalent_cache

LEAVE_V1 = ("The annual leave allowance is 21 days for every full-time employee. "
            "Unused leave expires at the end of March.")
LEAVE_V2 = ("The annual leave allowance is 25 days for every full-time employee. "
            "Unused leave expires at the end of March.")
TRAVEL = ("Business travel must be booked through the approved portal. "
          "Economy class applies to flights under 6 hours.")

_SENTENCES = [
    "The annual leave allowance is 21 days for every full-time employee.",
    "The annual leave allowance is 25 days for every full-time employee.",
    "Unused leave expires at the end of March.",
    "Business travel must be booked through the approved portal.",
    "Economy class applies to flights under 6 hours.",
]

QUERY = "How many days of annual leave do employees get?"


def _extractor(prompt: str) -> str:
    """A deterministic 'perfect extractor': claims = the known corpus sentences
    present in the prompt; cites every [S#] source block it was shown."""
    claims = [s for s in _SENTENCES if s in prompt]
    used = sorted({int(m) for m in re.findall(r"\[S(\d+)\]", prompt)})
    return json.dumps({
        "summary": "Extracted facts from the sources.",
        "claims": claims,
        "entities": ["annual leave", "business travel"],
        "facts": {},
        "used": used,
    })


@pytest.fixture()
def rig(embeddings: BagOfWordsEmbeddings) -> tuple[FakeVectorStore, ScriptedChatModel,
                                                   SemanticCache, CoalentRetriever,
                                                   list[dict[str, Any]]]:
    vs = FakeVectorStore(embeddings)
    vs.add_texts(
        [LEAVE_V1, TRAVEL],
        metadatas=[{"source": "policy:leave"}, {"source": "policy:travel"}],
    )
    chat = ScriptedChatModel(script=_extractor)
    events: list[dict[str, Any]] = []
    cache = create_coalent_cache(vs, llm=chat, embeddings=embeddings, on_event=events.append)
    return vs, chat, cache, CoalentRetriever(cache=cache), events


def test_lg2_cold_build_then_warm_hit(
    rig: tuple[FakeVectorStore, ScriptedChatModel, SemanticCache, CoalentRetriever,
               list[dict[str, Any]]],
) -> None:
    _vs, chat, _cache, retriever, _events = rig

    # -- cold: the miss builds understanding (LLM synthesis ran)
    cold = retriever.invoke(QUERY)
    assert len(cold) == 1
    doc = cold[0]
    assert doc.metadata["cache_hit"] is False
    assert doc.metadata["read_id"] == "read-1"
    assert "policy:leave" in doc.metadata["sources"]
    assert "21 days" in doc.page_content            # the answer fact is in the payload
    assert "[policy:leave]" in doc.page_content     # attributed (pool_header default)
    build_calls = len(chat.calls)
    assert build_calls >= 1

    # -- warm: the same question serves from cache with ZERO LLM calls
    warm = retriever.invoke(QUERY)[0]
    assert warm.metadata["cache_hit"] is True
    assert warm.metadata["read_id"] != doc.metadata["read_id"]
    assert "21 days" in warm.page_content
    assert len(chat.calls) == build_calls

    # -- paraphrase: still warm (matched by meaning, not string equality)
    para = retriever.invoke("What is the annual leave allowance?")[0]
    assert para.metadata["cache_hit"] is True
    assert "21 days" in para.page_content
    assert len(chat.calls) == build_calls


def test_lg2_include_evidence_returns_raw_chunks(
    rig: tuple[FakeVectorStore, ScriptedChatModel, SemanticCache, CoalentRetriever,
               list[dict[str, Any]]],
) -> None:
    _vs, _chat, cache, _retriever, _events = rig
    retriever = CoalentRetriever(cache=cache, include_evidence=True)
    docs = retriever.invoke(QUERY)
    assert len(docs) >= 2
    evidence = [d for d in docs if d.metadata.get("coalent_evidence")]
    assert evidence and any(d.metadata["source"] == "policy:leave" for d in evidence)
    assert any(LEAVE_V1 == d.page_content for d in evidence)


def test_lg3_source_changed_rebuilds_next_read(
    rig: tuple[FakeVectorStore, ScriptedChatModel, SemanticCache, CoalentRetriever,
               list[dict[str, Any]]],
) -> None:
    vs, chat, cache, retriever, events = rig

    # warm the cache on v1
    assert "21 days" in retriever.invoke(QUERY)[0].page_content
    warm_calls = len(chat.calls)

    # the user's ingestion pipeline updates the doc in THEIR index, then signals Coalent
    vs.replace_text("policy:leave", LEAVE_V2)
    invalidation = cache.source_changed("policy:leave", text=LEAVE_V2)
    assert invalidation.matched_units >= 1
    assert invalidation.dirtied                     # surgically dirtied, not wiped

    # the very next read rebuilds (fresh synthesis) and serves the NEW fact
    after = retriever.invoke(QUERY)[0]
    assert after.metadata["cache_hit"] is False
    assert "25 days" in after.page_content
    assert "21 days" not in after.page_content      # no stale claim survives
    assert len(chat.calls) > warm_calls
    assert any(e.get("event") == "stale_read_prevented" for e in events)

    # and the read after that is warm again, on the new content — zero further LLM spend
    calls_after_rebuild = len(chat.calls)
    settled = retriever.invoke(QUERY)[0]
    assert settled.metadata["cache_hit"] is True
    assert "25 days" in settled.page_content
    assert len(chat.calls) == calls_after_rebuild


def test_lg3_unchanged_content_is_a_noop(
    rig: tuple[FakeVectorStore, ScriptedChatModel, SemanticCache, CoalentRetriever,
               list[dict[str, Any]]],
) -> None:
    _vs, chat, cache, retriever, _events = rig
    retriever.invoke(QUERY)
    calls = len(chat.calls)
    result = cache.source_changed("policy:leave", text=LEAVE_V1)  # same content, same hash
    assert not result.dirtied
    assert result.skipped_unchanged                  # content-hash earns its keep
    assert retriever.invoke(QUERY)[0].metadata["cache_hit"] is True
    assert len(chat.calls) == calls
