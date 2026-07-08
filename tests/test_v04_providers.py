"""v0.4 — provider-side usage extraction (OpenAI/Anthropic), via injected fake clients (no SDK).

Exercises the branch-heavy mapping the shipped adapters do: usage present/absent, the getattr +
`or 0` guards for missing/None token fields, None content, and Anthropic's text-block filtering.
"""
from __future__ import annotations

from typing import Any

from coalent.adapters.providers import AnthropicProvider, OpenAIProvider
from coalent.semantic import Generation

_KW: dict[str, Any] = dict(model="m", system="s", user="u", max_tokens=10, temperature=0.0)


class _Attrs:
    """A bag whose attributes are whatever you pass — to omit a field, don't pass it."""

    def __init__(self, **attrs: Any) -> None:
        self.__dict__.update(attrs)


# ----------------------------------------------------------------- OpenAI
class _OACompletions:
    def __init__(self, resp: Any) -> None:
        self._resp = resp

    def create(self, **kw: Any) -> Any:
        return self._resp


class _OAClient:
    def __init__(self, resp: Any) -> None:
        self.chat = _Attrs(completions=_OACompletions(resp))


def _oa_resp(content: Any, usage: Any) -> Any:
    return _Attrs(choices=[_Attrs(message=_Attrs(content=content))], usage=usage)


def _openai(resp: Any) -> OpenAIProvider:
    return OpenAIProvider(client=_OAClient(resp))


def test_openai_maps_usage() -> None:
    g = _openai(_oa_resp("hello", _Attrs(prompt_tokens=100, completion_tokens=20))).generate(**_KW)
    assert isinstance(g, Generation)
    assert g.text == "hello"
    assert g.usage is not None
    assert (g.usage.prompt_tokens, g.usage.completion_tokens, g.usage.model) == (100, 20, "m")


def test_openai_no_usage_is_none() -> None:
    g = _openai(_oa_resp("hi", None)).generate(**_KW)
    assert g.usage is None


def test_openai_missing_or_none_token_fields_coerce_to_zero() -> None:
    # prompt_tokens is None (the `or 0` guard); completion_tokens absent (the getattr default)
    g = _openai(_oa_resp("hi", _Attrs(prompt_tokens=None))).generate(**_KW)
    assert g.usage is not None
    assert (g.usage.prompt_tokens, g.usage.completion_tokens) == (0, 0)


def test_openai_none_content_is_empty_text() -> None:
    g = _openai(_oa_resp(None, _Attrs(prompt_tokens=1, completion_tokens=1))).generate(**_KW)
    assert g.text == ""


# ----------------------------------------------------------------- Anthropic
class _AnMessages:
    def __init__(self, resp: Any) -> None:
        self._resp = resp

    def create(self, **kw: Any) -> Any:
        return self._resp


class _AnClient:
    def __init__(self, resp: Any) -> None:
        self.messages = _AnMessages(resp)


def _anth(resp: Any) -> AnthropicProvider:
    return AnthropicProvider(client=_AnClient(resp))


def test_anthropic_joins_text_blocks_and_maps_usage() -> None:
    resp = _Attrs(
        content=[
            _Attrs(type="tool_use", text="IGNORE"),  # non-text block dropped
            _Attrs(type="text", text="a"),
            _Attrs(type="text", text="b"),
        ],
        usage=_Attrs(input_tokens=30, output_tokens=7),
    )
    g = _anth(resp).generate(**_KW)
    assert g.text == "ab"
    assert g.usage is not None
    assert (g.usage.prompt_tokens, g.usage.completion_tokens, g.usage.model) == (30, 7, "m")


def test_anthropic_no_usage_is_none() -> None:
    g = _anth(_Attrs(content=[_Attrs(type="text", text="x")], usage=None)).generate(**_KW)
    assert g.usage is None
