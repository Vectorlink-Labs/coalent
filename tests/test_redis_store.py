"""Tests for RedisCognitionStore.

The store is bring-your-own-client by design, so we exercise it against a tiny
in-memory fake that implements just the redis surface it uses (get/set/delete/
scan_iter) — no live server required. Proves the store's own logic: key prefixing,
JSON round-trip, scan-based ``all()``/``__len__``, and restart-safe invalidation.
"""
from __future__ import annotations

import re
from typing import Iterator, Pattern

from coalent.semantic import (
    InMemoryRetriever,
    RedisCognitionStore,
    SemanticCache,
    StubSynthesizer,
)

HR = "confluence:hr"


def _glob_to_regex(pattern: str) -> Pattern[str]:
    """Faithfully emulate Redis SCAN glob: ``*`` ``?`` ``[...]`` and ``\\`` escaping —
    so the fake catches the bug a metacharacter prefix would cause on a real server."""
    out = ["^"]
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\" and i + 1 < len(pattern):
            out.append(re.escape(pattern[i + 1]))
            i += 2
            continue
        if ch == "*":
            out.append(".*")
        elif ch == "?":
            out.append(".")
        elif ch == "[":
            close = pattern.find("]", i + 1)
            if close != -1:
                out.append("[" + pattern[i + 1 : close] + "]")
                i = close + 1
                continue
            out.append(re.escape(ch))
        else:
            out.append(re.escape(ch))
        i += 1
    out.append("$")
    return re.compile("".join(out))


class FakeRedis:
    """Just enough of the redis client surface for RedisCognitionStore."""

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}

    def get(self, key: str) -> bytes | None:
        return self._data.get(key)

    def set(self, key: str, value: str) -> None:
        self._data[key] = value.encode("utf-8")  # mirror real redis: bytes back out

    def delete(self, *keys: str) -> None:
        for key in keys:
            self._data.pop(key, None)

    def scan_iter(self, match: str = "*") -> Iterator[str]:
        regex = _glob_to_regex(match)
        return iter([key for key in list(self._data) if regex.match(key)])


def _retriever() -> InMemoryRetriever:
    retriever = InMemoryRetriever()
    retriever.add(HR, "leave policy: 21 days of annual leave per year")
    return retriever


def test_put_get_delete_roundtrip() -> None:
    store = RedisCognitionStore(FakeRedis())
    cache = SemanticCache(_retriever(), StubSynthesizer(), store=store)

    result = cache.get("leave policy")
    assert len(store) == 1

    fetched = store.get(result.unit_id)
    assert fetched is not None
    assert fetched.id == result.unit_id

    store.delete(result.unit_id)
    assert store.get(result.unit_id) is None
    assert len(store) == 0


def test_key_prefix_isolation() -> None:
    client = FakeRedis()
    store = RedisCognitionStore(client, prefix="acme:cog:")
    SemanticCache(_retriever(), StubSynthesizer(), store=store).get("leave policy")

    assert len(store) == 1
    assert all(key.startswith("acme:cog:") for key in client.scan_iter("acme:cog:*"))


def test_units_and_invalidation_survive_restart() -> None:
    client = FakeRedis()  # the shared "server" across two cache instances

    store1 = RedisCognitionStore(client)
    cache1 = SemanticCache(_retriever(), StubSynthesizer(), store=store1)
    first = cache1.get("what is our leave policy")
    assert first.cache_hit is False

    # restart: brand-new store + cache over the same backing redis
    store2 = RedisCognitionStore(client)
    cache2 = SemanticCache(_retriever(), StubSynthesizer(), store=store2)
    second = cache2.get("what is our leave policy")
    assert second.cache_hit is True
    assert second.unit_id == first.unit_id

    # invalidation still fires after restart (indexes rebuilt from the store)
    changed = cache2.source_changed(HR, text="leave policy: now 25 days")
    assert first.unit_id in changed.dirtied


def test_scan_duplicates_are_deduped_in_len_and_all() -> None:
    class DupRedis(FakeRedis):
        # real Redis SCAN may yield a key more than once (rehash / concurrent writes)
        def scan_iter(self, match: str = "*") -> Iterator[str]:
            keys = list(super().scan_iter(match))
            return iter(keys + keys)

    store = RedisCognitionStore(DupRedis())
    SemanticCache(_retriever(), StubSynthesizer(), store=store).get("leave policy")
    assert len(store) == 1          # not 2
    assert len(store.all()) == 1    # every unit, once


def test_glob_metacharacter_prefix_is_escaped_in_scan() -> None:
    store = RedisCognitionStore(FakeRedis(), prefix="acme[prod]:cog:")
    # the brackets are escaped so SCAN matches the LITERAL prefix, not a char class
    assert "\\[prod\\]" in store._match()

    # end-to-end: writing + scanning agree, so a restart rebuilds (not empty)
    client = FakeRedis()
    first = SemanticCache(
        _retriever(), StubSynthesizer(), store=RedisCognitionStore(client, prefix="acme[prod]:cog:")
    ).get("leave policy")
    store2 = RedisCognitionStore(client, prefix="acme[prod]:cog:")
    assert len(store2) == 1
    assert store2.get(first.unit_id) is not None


def test_poisoned_key_does_not_brick_the_restart_rebuild() -> None:
    client = FakeRedis()
    good = SemanticCache(
        _retriever(), StubSynthesizer(), store=RedisCognitionStore(client)
    ).get("leave policy")

    # another process writes garbage under the shared prefix
    client.set("coalent:cognition:garbage", "not json at all")

    # restart: the bad key is skipped, the good unit still rebuilds (no crash)
    cache2 = SemanticCache(_retriever(), StubSynthesizer(), store=RedisCognitionStore(client))
    assert cache2.get("what is our leave policy").unit_id == good.unit_id
