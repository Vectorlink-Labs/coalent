"""v0.4 — the EXTRACT GATE, pinned (REAL OpenAI, skipped offline).

The distilled, cheap regression behind bench_extract_gate.py: it exercises the SHIPPED
``LLMSynthesizer(extract=True)`` path on real embeddings + a real model and pins the two
claims that justify the extractive pivot, with ONE synthesis call and NO judge loop:

  1. the cached atoms capture the document's numbers (query-INDEPENDENT fact_recall is high),
     so one unit can answer many later number questions;
  2. a genuinely ABSENT fact ESCALATES (the RAG floor fires) instead of confidently serving a
     wrong number — the cache never silently degrades below retrieval.

The full ablation vs prose and the answer-correctness judging live in bench_extract_gate.py
(run manually with a key). Keeping this focused keeps the pinned regression's token cost low.
"""
from __future__ import annotations

import importlib.util
import os
import re

import pytest

pytestmark = pytest.mark.openai

# Both the key AND the SDK: with a key exported but the openai package absent
# (plain `pip install -e ".[dev]"`), these must SKIP, not crash on import.
_needs_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY") or importlib.util.find_spec("openai") is None,
    reason="the extract gate needs a real OPENAI_API_KEY and the openai SDK installed",
)

# A compact structured policy doc — the kind of reuse-heavy source the cache wins on.
CHUNKS = [
    ("hr:annual", "Full-time employees accrue 20 days of paid annual leave per year. Up to 5 unused "
     "days may be carried over; beyond 5 is forfeited on December 31."),
    ("hr:sick", "Employees receive 10 paid sick days per year. A doctor's note is required for any "
     "absence longer than 3 consecutive days. Care for an immediate family member is capped at 5 days."),
    ("hr:parental", "Primary caregivers get 16 weeks of paid parental leave; secondary caregivers get "
     "6 weeks. It must be taken within 12 months of birth or adoption."),
    ("hr:other", "Bereavement leave is 5 days for immediate family. Jury duty is paid up to 10 days. "
     "There are 12 public holidays per year."),
]
GOLD = [
    ("20", "annual"), ("5", "carry"), ("10", "sick"), ("3", "doctor"), ("5", "family"),
    ("16", "primary"), ("6", "secondary"), ("12", "months"), ("5", "bereavement"),
    ("10", "jury"), ("12", "public"),
]
PRESENT = ("how many paid sick days per year?", "10")
ABSENT = [
    "how many days of paid menstrual leave?",
    "what is the pet bereavement leave allowance?",
]


def _has(value: str, text: str) -> bool:
    """value appears as a standalone number (not a substring of a larger number)."""
    return re.search(rf"(?<!\d){re.escape(value)}(?!\d)", text) is not None


def _utext(u: dict) -> str:
    parts = [str(u.get("summary", ""))] + [str(c) for c in (u.get("claims") or [])]
    facts = u.get("facts")
    if isinstance(facts, dict):
        parts += [f"{k}: {v}" for k, v in facts.items()]
    return "\n".join(p for p in parts if p)


def _extractive_cache():  # type: ignore[no-untyped-def]
    from coalent import Chunk, LLMSynthesizer, OpenAIProvider, SemanticCache
    from coalent.semantic import FunctionRetriever, OpenAIEmbedder

    cs = [Chunk(artifact_id=a, text=t) for a, t in CHUNKS]
    cache = SemanticCache(
        FunctionRetriever(lambda q, ns: list(cs)),
        LLMSynthesizer(OpenAIProvider(), model="gpt-4o-mini", extract=True, max_tokens=1200),
        embedder=OpenAIEmbedder(), hit_threshold=0.0, coverage_floor=0.4, read_path="unit",
        enable_coverage_escalation=True,
    )
    cache.get("summarize this document")   # neutral build query -> the ONE extractive unit
    return cache


@_needs_key
def test_extractive_atoms_capture_the_documents_numbers() -> None:
    cache = _extractive_cache()
    unit = next(iter(cache._units.values()))
    text = _utext(unit.understanding)
    low = text.lower()
    hit = sum(1 for value, what in GOLD if _has(value, text) and what in low)
    recall = hit / len(GOLD)
    assert recall >= 0.85, f"extractive fact_recall too low: {recall:.0%}\n{text}"


@_needs_key
def test_present_fact_is_served_from_the_cache() -> None:
    cache = _extractive_cache()
    q, gold = PRESENT
    res = cache.get(q)
    served = _utext(res.context.get("understanding", {}))
    if res.escalated:
        served += "\n\n" + res.raw_text
    assert _has(gold, served)              # the number is available to the answerer


@_needs_key
def test_absent_fact_escalates_to_the_rag_floor() -> None:
    cache = _extractive_cache()
    escalated = sum(int(cache.get(q).escalated) for q in ABSENT)
    # A fact the doc does NOT contain must fall to fresh retrieval, not be confidently served.
    assert escalated >= len(ABSENT) - 1, "absent facts should escalate (the RAG floor must fire)"
