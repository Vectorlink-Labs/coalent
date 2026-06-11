"""LLMSynthesizer — structured, citation-grounded understanding (the envelope).

**You own the content, Coalent owns the envelope.** You decide *what* understanding
to produce (the ``instruction`` and optional ``fields``); Coalent always wraps it
with the candidate sources, a strict-JSON contract, and a ``used`` citation list —
so provenance is captured no matter what you ask for. Only the sources the model
cites can invalidate the unit. A parse failure degrades (``ok=False``) instead of
caching fabricated text.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from .ports import Chunk, LLMProvider, Synthesis

#: Coalent-owned. Always present — the role, grounding rules, and output format.
_SYSTEM = (
    "You are a meticulous analyst who turns source material into decision-ready "
    "understanding. Work ONLY from the provided sources — never invent facts or use "
    "outside knowledge. Reason across the sources to connect related information. If "
    "sources disagree, prefer the most recent or most authoritative and note the "
    "conflict. Cite the exact sources you used. Respond with STRICT JSON only — no "
    "prose, no markdown, no code fences."
)

#: User-owned (overridable). What understanding to produce.
_DEFAULT_INSTRUCTION = (
    "Produce a faithful, decision-ready understanding of the question: a concise "
    "summary of what actually matters for a decision, the key claims (each grounded "
    "in a source), the salient entities, and the concrete facts (names, numbers, "
    "dates, statuses). Be specific and tight — no padding."
)

#: An instruction is either a fixed string or computed per query.
InstructionFn = Callable[[str], str]

_DEFAULT_FIELDS = ["summary", "claims", "entities", "facts"]


def _build_prompt(query: str, chunks: list[Chunk], instruction: str, fields: list[str]) -> str:
    lines = ['SOURCES (cite the [S#] indices you actually use in "used"):']
    for index, chunk in enumerate(chunks):
        lines.append(f"[S{index}] ({chunk.artifact_id})\n{chunk.text}")
    lines.append(f"\nQUESTION: {query}\n")
    lines.append(instruction)  # <- user-owned content
    spec = ", ".join(f'"{name}"' for name in fields)
    lines.append(
        f'Return ONLY a JSON object with these keys: {spec}, and "used" '
        "(the list of integer S-indices you actually relied on)."
    )
    return "\n".join(lines)


def _parse_json(text: str) -> dict[str, Any] | None:
    """Best-effort strict-JSON extraction, tolerant of surrounding prose."""
    stripped = text.strip()
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            obj = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None


def _coerce_indices(value: Any, count: int) -> list[int]:
    """Keep only valid, in-range, de-duplicated integer indices."""
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        if isinstance(item, bool):
            continue
        if isinstance(item, int) and 0 <= item < count and item not in out:
            out.append(item)
    return out


class LLMSynthesizer:
    """A citation-grounded :class:`~coalent.semantic.ports.Synthesizer`.

    ``instruction`` (a string, or a ``query -> str`` callable) and ``fields`` are
    yours to define — Coalent wraps them in the source/JSON/citation envelope.
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        model: str = "gpt-4o-mini",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        retries: int = 1,
        instruction: str | InstructionFn = _DEFAULT_INSTRUCTION,
        fields: list[str] | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._retries = retries
        self._instruction = instruction
        self._fields = fields or _DEFAULT_FIELDS

    def _instruction_for(self, query: str) -> str:
        instruction = self._instruction
        return instruction(query) if callable(instruction) else instruction

    def synthesize(self, query: str, chunks: list[Chunk]) -> Synthesis:
        prompt = _build_prompt(query, chunks, self._instruction_for(query), self._fields)
        parsed = self._call(prompt)
        attempts = 0
        while parsed is None and attempts < self._retries:
            attempts += 1
            parsed = self._call(prompt + "\n\nReturn STRICT JSON only.")
        if parsed is None:
            # Total failure -> degrade. Never fabricate or smuggle raw text in.
            return Synthesis(understanding={"_synthesis_failed": True}, used=[], ok=False)
        used = _coerce_indices(parsed.get("used"), len(chunks))
        understanding = {key: value for key, value in parsed.items() if key != "used"}
        return Synthesis(understanding=understanding, used=used, ok=True)

    def _call(self, prompt: str) -> dict[str, Any] | None:
        text = self._provider.generate(
            model=self._model,
            system=_SYSTEM,
            user=prompt,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        return _parse_json(text)


#: Sentinel distinguishing "not JSON" from a genuine JSON ``null`` (-> ``None``).
_UNPARSED: Any = object()


def _loads(text: str) -> Any:
    """Parse text as JSON, or return ``_UNPARSED`` if it isn't JSON.

    A literal ``null`` parses to ``None`` (a structured value), not ``_UNPARSED`` —
    so the synthesizer keeps a real null as a value, not as the raw string.
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return _UNPARSED


class JSONPassthroughSynthesizer:
    """Cache already-structured tool/API output as-is — the JSON *is* the understanding.

    **No LLM call.** A REST/MCP tool result is already decision-ready, so paying a
    model to "summarize" it adds latency and cost for nothing. This synthesizer parses
    each chunk's text as JSON and folds it straight into the understanding: objects go
    into ``facts``, other JSON values and non-JSON text become ``claims``. Every chunk
    is cited (``used`` = all), so provenance and :class:`FreshnessPolicy` revalidation
    still work — a changed tool result invalidates the unit exactly like a doc edit.

    Best for single-result tool caches. When several chunks carry objects with
    overlapping keys they merge (later wins); pass ``merge=False`` to keep each record
    as its own claim instead. Provide ``summary`` to set a fixed headline (otherwise a
    plain record count is used — never a fabricated description).
    """

    def __init__(self, *, summary: str | None = None, merge: bool = True) -> None:
        self._summary = summary
        self._merge = merge

    def synthesize(self, query: str, chunks: list[Chunk]) -> Synthesis:
        facts: dict[str, Any] = {}
        claims: list[Any] = []
        for chunk in chunks:
            value = _loads(chunk.text)
            if value is _UNPARSED:
                claims.append(chunk.text)   # not JSON -> keep the raw line as a claim
            elif isinstance(value, dict) and self._merge:
                facts.update(value)
            else:
                claims.append(value)        # list / scalar / null / (unmerged) object
        summary = self._summary if self._summary is not None else f"{len(chunks)} record(s) from source."
        # ``_passthrough`` tells the cache's projection NOT to query-trim this record:
        # a tool result is already minimal, so every field must survive into context.
        understanding: dict[str, Any] = {
            "summary": summary, "facts": facts, "claims": claims, "_passthrough": True,
        }
        return Synthesis(understanding=understanding, used=list(range(len(chunks))), ok=True)
