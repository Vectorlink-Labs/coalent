"""LLM providers.

Concrete implementations of the :class:`~coalent.semantic.ports.LLMProvider`
port. The real providers import their SDKs lazily, so the base package installs
without them — pull them in via the ``openai`` / ``anthropic`` extras.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any


class StubProvider:
    """Deterministic, network-free provider for development and tests.

    With no ``canned`` response it returns valid JSON that cites every candidate
    span found in the prompt (simulating "the model read and used everything").
    Pass ``canned`` to script an exact response (e.g. to test selective
    citation or malformed output).
    """

    def __init__(self, *, canned: str | None = None) -> None:
        self._canned = canned
        self.calls: list[dict[str, Any]] = []  # recorded for assertions

    def generate(
        self, *, model: str, system: str, user: str, max_tokens: int, temperature: float
    ) -> str:
        self.calls.append({"model": model, "max_tokens": max_tokens})
        if self._canned is not None:
            return self._canned
        keys = re.findall(r"key=(\S+)", user)
        return json.dumps(
            {"summary": f"[stub:{model}] decision-ready understanding", "used_span_keys": keys}
        )


class OpenAIProvider:
    """Thin adapter over the OpenAI Chat Completions API (``openai`` extra)."""

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None) -> None:
        from openai import OpenAI  # lazy: only required when actually used

        self._client: Any = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"), base_url=base_url
        )

    def generate(
        self, *, model: str, system: str, user: str, max_tokens: int, temperature: float
    ) -> str:
        response = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = response.choices[0].message.content
        return content if isinstance(content, str) else ""


class AnthropicProvider:
    """Thin adapter over the Anthropic Messages API (``anthropic`` extra)."""

    def __init__(self, *, api_key: str | None = None) -> None:
        import anthropic  # lazy

        self._client: Any = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    def generate(
        self, *, model: str, system: str, user: str, max_tokens: int, temperature: float
    ) -> str:
        response = self._client.messages.create(
            model=model,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": user}],
        )
        parts = [
            getattr(block, "text", "")
            for block in response.content
            if getattr(block, "type", None) == "text"
        ]
        return "".join(str(part) for part in parts)
