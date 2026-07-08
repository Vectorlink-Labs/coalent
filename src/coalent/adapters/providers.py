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

from ..semantic.ports import Generation, Usage


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

    def __init__(
        self, *, api_key: str | None = None, base_url: str | None = None, client: Any = None
    ) -> None:
        if client is None:
            from openai import OpenAI  # lazy: only required when actually used

            client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"), base_url=base_url)
        self._client: Any = client  # inject your own client (or a fake) to bypass the SDK

    def generate(
        self, *, model: str, system: str, user: str, max_tokens: int, temperature: float
    ) -> Generation:
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
        text = content if isinstance(content, str) else ""
        raw = getattr(response, "usage", None)
        usage = (
            Usage(
                prompt_tokens=int(getattr(raw, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(raw, "completion_tokens", 0) or 0),
                model=model,
            )
            if raw is not None
            else None
        )
        return Generation(text=text, usage=usage)


class AnthropicProvider:
    """Thin adapter over the Anthropic Messages API (``anthropic`` extra)."""

    def __init__(self, *, api_key: str | None = None, client: Any = None) -> None:
        if client is None:
            import anthropic  # lazy

            client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self._client: Any = client  # inject your own client (or a fake) to bypass the SDK

    def generate(
        self, *, model: str, system: str, user: str, max_tokens: int, temperature: float
    ) -> Generation:
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
        text = "".join(str(part) for part in parts)
        raw = getattr(response, "usage", None)
        usage = (
            Usage(
                prompt_tokens=int(getattr(raw, "input_tokens", 0) or 0),
                completion_tokens=int(getattr(raw, "output_tokens", 0) or 0),
                model=model,
            )
            if raw is not None
            else None
        )
        return Generation(text=text, usage=usage)
