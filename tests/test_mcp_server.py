"""The MCP server gauntlet — offline-deterministic.

v1 (MCP-SPEC.md, locked 2026-08-03): G1 protocol · G2 cold→warm · G3 the freshness
bar · G4 the behavioral loop · G5 persistence · G6 fail-loud. v1.1
(INTEGRATIONS-SPEC.md Task 1): the factory-mode (BYO retriever) mirror of G2–G5,
the ``source_changed`` round-trip in both modes, the http two-concurrent-clients
smoke, and the bearer-auth guard. All bars run against the REAL server object
through the SDK's transports (in-process memory streams for stdio-shaped tests, a
real localhost streamable-HTTP server for the http bars — never a subprocess), with
a deterministic axis embedder + stub synthesizers standing in for OpenAI (per the
skip-offline directive, these gate PLUMBING, not router quality — the real-OpenAI
smoke happens before merge).

Tests that need the SDK carry ``needs_mcp`` so the zero-dep CI leg stays green;
``coalent.mcp`` itself imports cleanly without the extra (the import is lazy), so the
fail-loud G6 bars, the factory-loader units, and the corpus/chunker units run
everywhere.
"""
from __future__ import annotations

import asyncio
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from coalent import (
    FunctionEmbedder,
    InMemoryRetriever,
    SQLiteCognitionStore,
    StubSynthesizer,
)
from coalent.mcp import (
    CoalentMCP,
    WatchedCorpus,
    chunk_paragraphs,
    load_cache_factory,
    main,
)
from coalent.semantic import Chunk, SemanticCache, Synthesis

try:
    from mcp.client import Client
    HAS_MCP = True
except ImportError:  # zero-dep leg: the extra is not installed
    HAS_MCP = False

needs_mcp = pytest.mark.skipif(not HAS_MCP, reason="mcp extra not installed (zero-dep leg)")

# ---------------------------------------------------------------- offline determinism
# The v0.6 test geometry (mirrors tests/test_v06_residual_spans.py): axis-membership
# embeddings make every cosine in the gauntlet a hand-checkable fraction.
_AXES = ("alpha", "beta", "gamma", "kappa", "value", "junk")
_STRIP = re.compile(r"[^\w\s]")
_SENT = re.compile(r"(?<=[.!?])\s+")


def _embed(text: str) -> list[float]:
    words = set(_STRIP.sub(" ", text.lower()).split())
    v = [1.0 if a in words else 0.0 for a in _AXES]
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v] if n else v


class _SentClaimSynth:
    """Deterministic stand-in for the extractive LLM: claims = every sentence."""

    def __init__(self) -> None:
        self.calls = 0

    def synthesize(self, query: str, chunks: list[Chunk]) -> Synthesis:
        self.calls += 1
        claims: list[str] = []
        for chunk in chunks:
            for sent in _SENT.split(chunk.text):
                s = sent.strip()
                if s and s not in claims:
                    claims.append(s)
        return Synthesis(understanding={"claims": claims}, used=list(range(len(chunks))))


class _DropNumberSynth(_SentClaimSynth):
    """An extraction that PROVABLY drops every number-bearing sentence — the measured
    extraction-loss shape the residual-span net exists for (G4's precondition)."""

    def synthesize(self, query: str, chunks: list[Chunk]) -> Synthesis:
        out = super().synthesize(query, chunks)
        claims = [c for c in out.understanding["claims"] if not re.search(r"\d", c)]
        return Synthesis(understanding={"claims": claims}, used=list(out.used))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_app(docs: Path, store: Path, synth: Any) -> CoalentMCP:
    return CoalentMCP(
        docs, budget=1000, store_path=store,
        embedder=FunctionEmbedder(_embed), synthesizer=synth,
    )


class _DictRetriever:
    """The BYO stand-in: a mutable word-overlap 'vector DB' the user's ingestion
    pipeline keeps current (mirrors the library tests' _WordRetriever shape)."""

    def __init__(self) -> None:
        self.docs: dict[str, str] = {}

    def set(self, artifact_id: str, text: str) -> None:
        self.docs[artifact_id] = text

    def retrieve(self, query: str, *, namespace: str | None = None) -> list[Chunk]:
        qs = set(_STRIP.sub(" ", query.lower()).split())
        return [Chunk(artifact_id=aid, text=text)
                for aid, text in sorted(self.docs.items())
                if qs & set(_STRIP.sub(" ", text.lower()).split())]


def _factory_cache(retriever: Any, synth: Any, store: Any = None) -> SemanticCache:
    """What a user's --cache-factory function returns: a fully self-configured cache
    (their retriever, embedder, header, store — every knob theirs)."""
    return SemanticCache(
        retriever, synth,
        embedder=FunctionEmbedder(_embed),
        read_path="pool", residual_spans=True, query_keys=True,
        pool_header=lambda u: f"[{u.evidence[0].artifact_id}]" if u.evidence else "##",
        store=store,
    )


async def _tool(client: Any, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    result = await client.call_tool(name, args or {})
    assert not result.is_error, f"{name} errored: {result.content}"
    payload = result.structured_content
    assert isinstance(payload, dict)
    return payload


_HEADER = r"\[{path} \| modified \d{{4}}-\d{{2}}-\d{{2}}\]"


# ------------------------------------------------------------------------ G1 protocol

@needs_mcp
def test_g1_protocol_six_tools_valid_schemas(tmp_path: Path) -> None:
    _write(tmp_path / "docs" / "a.md", "Alpha value reached 42 units in June.")
    app = _make_app(tmp_path / "docs", tmp_path / "state.db", _SentClaimSynth())

    async def flow() -> None:
        async with Client(app.server) as client:
            tools = {t.name: t for t in (await client.list_tools()).tools}
            assert set(tools) == {
                "get_context", "report_refusal", "report_success", "source_changed",
                "list_sources", "cache_stats", "refresh",
            }
            gc = tools["get_context"]
            assert gc.input_schema["required"] == ["query"]
            assert gc.input_schema["properties"]["query"]["type"] == "string"
            assert "budget" in gc.input_schema["properties"]   # optional int
            for name in ("report_refusal", "report_success"):
                assert tools[name].input_schema["required"] == ["read_id"]
            sc = tools["source_changed"]                       # v1.1: the BYO signal
            assert sc.input_schema["required"] == ["artifact_id"]
            assert "text" in sc.input_schema["properties"]     # optional new content
            for name in ("list_sources", "cache_stats", "refresh"):
                assert tools[name].input_schema.get("required", []) == []
            for tool in tools.values():
                assert tool.description             # descriptions are the marketing surface
                assert tool.output_schema is not None
                assert tool.output_schema["type"] == "object"
            # The freshness guarantee is stated PLAINLY on the read tool (spec).
            assert "invalidated" in (gc.description or "")
            # Bad input fails loud through the protocol, not silently.
            bad = await client.call_tool("get_context", {"query": "   "})
            assert bad.is_error
            bad2 = await client.call_tool("get_context", {"query": "alpha", "budget": -5})
            assert bad2.is_error

    asyncio.run(flow())
    app.close()


# --------------------------------------------------------------------- G2 cold → warm

@needs_mcp
def test_g2_cold_build_then_warm_hit_attributed(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    _write(docs / "a.md", "Alpha value reached 42 units in June.")
    _write(docs / "b.md", "Gamma value reached 77 units in July.")
    synth = _SentClaimSynth()
    app = _make_app(docs, tmp_path / "state.db", synth)

    async def flow() -> None:
        async with Client(app.server) as client:
            cold = await _tool(client, "get_context", {"query": "alpha gamma value"})
            assert cold["cache_hit"] is False
            assert "42" in cold["context"] and "77" in cold["context"]
            # ATTRIBUTED: the auto pool_header heads EVERY served group with file metadata.
            for path in ("a.md", "b.md"):
                assert re.search(_HEADER.format(path=path), cold["context"])
            assert set(cold["sources"]) == {"a.md", "b.md"}
            assert cold["read_id"]
            assert synth.calls == 2                 # one source-anchored build per file
            warm = await _tool(client, "get_context", {"query": "alpha gamma value"})
            assert warm["cache_hit"] is True
            assert synth.calls == 2                 # warm serve = ZERO synthesis calls
            assert "42" in warm["context"] and "77" in warm["context"]
            # Per-call budget override serves less and always restores the default.
            tiny = await _tool(client, "get_context",
                               {"query": "alpha gamma value", "budget": 8})
            assert len(tiny["context"]) <= len(warm["context"])
            assert app.cache._serve_budget == 1000

    asyncio.run(flow())
    app.close()


# ------------------------------------------------------------- G3 the freshness bar

@needs_mcp
def test_g3_edit_reflected_on_very_next_read(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    _write(docs / "a.md", "Alpha value reached 42 units in June.")
    synth = _SentClaimSynth()
    app = _make_app(docs, tmp_path / "state.db", synth)

    async def flow() -> None:
        async with Client(app.server) as client:
            cold = await _tool(client, "get_context", {"query": "alpha value"})
            assert "42" in cold["context"]
            warm = await _tool(client, "get_context", {"query": "alpha value"})
            assert warm["cache_hit"] is True        # cached — the edit must beat THIS
            stats0 = await _tool(client, "cache_stats")
            assert stats0["staleness_prevented"] == 0
            # Edit the watched file between calls (different size → stat-detectable
            # even under coarse filesystem mtime granularity).
            _write(docs / "a.md", "Alpha value reached 99 units in June, revised.")
            nxt = await _tool(client, "get_context", {"query": "alpha value"})
            # Byte-honest: the very next read serves the NEW content, not the cache.
            assert "99" in nxt["context"]
            assert "42" not in nxt["context"]
            assert nxt["cache_hit"] is False        # the read rebuilt instead of serving stale
            stats1 = await _tool(client, "cache_stats")
            assert stats1["staleness_prevented"] == 1

    asyncio.run(flow())
    app.close()


# ----------------------------------------------------------- G4 the behavioral loop

_G4_DOC = ("Alpha beta value coverage improved. "
           "Beta shipments hit 900 units in kappa region.")
_G4_MISSED = "Beta shipments hit 900 units in kappa region."


@needs_mcp
def test_g4_refusal_success_key_fires(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    _write(docs / "notes.md", _G4_DOC)
    app = _make_app(docs, tmp_path / "state.db", _DropNumberSynth())

    async def flow() -> None:
        async with Client(app.server) as client:
            built = await _tool(client, "get_context", {"query": "alpha beta value"})
            assert "900" not in built["context"]    # the extraction provably dropped the fact
            r1 = await _tool(client, "get_context", {"query": "beta value"})
            assert r1["cache_hit"] is True
            assert "900" not in r1["context"]       # sibling masking: rank stays confident
            # FORCED refusal → the span retry payload, attributed under the file header.
            ref = await _tool(client, "report_refusal", {"read_id": r1["read_id"]})
            assert ref["payload"] is not None
            assert "[source excerpt] " + _G4_MISSED in ref["payload"]
            assert re.search(_HEADER.format(path="notes.md"), ref["payload"])
            # Unknown read_id: null payload, never an error.
            bogus = await _tool(client, "report_refusal", {"read_id": "bogus"})
            assert bogus["payload"] is None
            # The retry worked → confirm; the key becomes durable (idempotent after).
            suc = await _tool(client, "report_success", {"read_id": r1["read_id"]})
            assert suc["confirmed"] is True
            again = await _tool(client, "report_success", {"read_id": r1["read_id"]})
            assert again["confirmed"] is False
            # A mild paraphrase of the refused query now FIRES the key: the missed fact
            # serves in the payload (cosine paraphrase↔key = 1.0 ≥ key_floor 0.85 under
            # the axis embedder — deterministic, so the fire is asserted, not hoped for).
            fired = await _tool(client, "get_context", {"query": "value beta"})
            assert "900" in fired["context"]

    asyncio.run(flow())
    app.close()


# --------------------------------------------------------------------- G5 persistence

@needs_mcp
def test_g5_store_survives_restart(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    _write(docs / "notes.md", _G4_DOC)
    store = tmp_path / "state" / "store.db"
    synth1 = _DropNumberSynth()
    app1 = _make_app(docs, store, synth1)

    async def session_one() -> None:
        async with Client(app1.server) as client:
            await _tool(client, "get_context", {"query": "alpha beta value"})
            r1 = await _tool(client, "get_context", {"query": "beta value"})
            ref = await _tool(client, "report_refusal", {"read_id": r1["read_id"]})
            assert ref["payload"] is not None
            suc = await _tool(client, "report_success", {"read_id": r1["read_id"]})
            assert suc["confirmed"] is True

    asyncio.run(session_one())
    app1.close()                                    # kill the server

    synth2 = _DropNumberSynth()
    app2 = _make_app(docs, store, synth2)           # restart on the same store

    async def session_two() -> None:
        async with Client(app2.server) as client:
            stats = await _tool(client, "cache_stats")
            assert stats["units"] == 1              # the unit loaded from the store
            warm = await _tool(client, "get_context", {"query": "alpha beta value"})
            assert warm["cache_hit"] is True        # warm hit, no rebuild after restart
            assert synth2.calls == 0                # zero synthesis in the new process
            # The CONFIRMED key survived serde: the paraphrase still fires next session.
            fired = await _tool(client, "get_context", {"query": "value beta"})
            assert "900" in fired["context"]

    asyncio.run(session_two())
    app2.close()


# ---------------------------------------------- corpus lifecycle through the tools

@needs_mcp
def test_refresh_and_list_sources_lifecycle(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    _write(docs / "a.md", "Alpha value reached 42 units in June.")
    synth = _SentClaimSynth()
    app = _make_app(docs, tmp_path / "state.db", synth)

    async def flow() -> None:
        async with Client(app.server) as client:
            await _tool(client, "get_context", {"query": "alpha value"})
            # A NEW file appears; list_sources (read-only) sees it pending…
            _write(docs / "c.md", "Gamma kappa throughput doubled overnight, they say.")
            listed = await _tool(client, "list_sources")
            states = {s["path"]: s["state"] for s in listed["sources"]}
            assert states == {"a.md": "fresh", "c.md": "new-pending"}
            assert all(s["last_modified"] for s in listed["sources"])
            # …refresh ingests it…
            ref = await _tool(client, "refresh")
            assert ref["added"] == ["c.md"]
            assert ref["changed"] == [] and ref["removed"] == []
            # …and a read on its topic builds from it.
            got = await _tool(client, "get_context", {"query": "gamma kappa"})
            assert "doubled overnight" in got["context"]
            assert "c.md" in got["sources"]
            # A DELETED file evicts its units on the next scan.
            (docs / "a.md").unlink()
            listed2 = await _tool(client, "list_sources")
            states2 = {s["path"]: s["state"] for s in listed2["sources"]}
            assert states2["a.md"] == "removed-pending"
            ref2 = await _tool(client, "refresh")
            assert ref2["removed"] == ["a.md"]
            assert ref2["units_evicted"] == 1
            stats = await _tool(client, "cache_stats")
            assert stats["units"] == 1              # only c.md's unit remains

    asyncio.run(flow())
    app.close()


# --------------------------------------- v1.1 factory (BYO) mode: the G2-G5 mirror

@needs_mcp
def test_factory_cold_warm_and_byo_freshness(tmp_path: Path) -> None:
    # F2+F3: the G2/G3 bars driven through a BYO retriever — no folder, no scan;
    # freshness arrives through the source_changed tool (the user's pipeline).
    ret = _DictRetriever()
    ret.set("kb:alpha", "Alpha value reached 42 units in June.")
    synth = _SentClaimSynth()
    app = CoalentMCP(cache=_factory_cache(ret, synth))

    async def flow() -> None:
        async with Client(app.server) as client:
            cold = await _tool(client, "get_context", {"query": "alpha value"})
            assert cold["cache_hit"] is False
            assert "42" in cold["context"]
            assert "[kb:alpha]" in cold["context"]   # the FACTORY's header attributes
            assert cold["sources"] == ["kb:alpha"]
            warm = await _tool(client, "get_context", {"query": "alpha value"})
            assert warm["cache_hit"] is True and synth.calls == 1
            listed = await _tool(client, "list_sources")
            assert listed["watch_dir"] is None and listed["sources"] == []
            ref = await _tool(client, "refresh")     # no-op shape without watching
            assert ref == {"added": [], "changed": [], "removed": [],
                           "units_dirtied": 0, "units_evicted": 0}
            # Skip-no-op: signalling UNCHANGED content is hash-detected, no dirty.
            same = await _tool(client, "source_changed", {
                "artifact_id": "kb:alpha",
                "text": "Alpha value reached 42 units in June."})
            assert same["matched_units"] == 1
            assert same["dirtied"] == [] and len(same["skipped_unchanged"]) == 1
            # The BYO freshness bar: pipeline updates its index, then signals.
            ret.set("kb:alpha", "Alpha value reached 99 units in June, revised.")
            sc = await _tool(client, "source_changed", {
                "artifact_id": "kb:alpha",
                "text": "Alpha value reached 99 units in June, revised."})
            assert sc["matched_units"] == 1 and len(sc["dirtied"]) == 1
            nxt = await _tool(client, "get_context", {"query": "alpha value"})
            assert "99" in nxt["context"] and "42" not in nxt["context"]
            assert nxt["cache_hit"] is False
            stats = await _tool(client, "cache_stats")
            assert stats["staleness_prevented"] == 1

    asyncio.run(flow())


@needs_mcp
def test_factory_behavioral_loop(tmp_path: Path) -> None:
    # F4: the G4 bar through a BYO retriever — refusal → success → key fires.
    ret = _DictRetriever()
    ret.set("kb:notes", _G4_DOC)
    app = CoalentMCP(cache=_factory_cache(ret, _DropNumberSynth()))

    async def flow() -> None:
        async with Client(app.server) as client:
            await _tool(client, "get_context", {"query": "alpha beta value"})
            r1 = await _tool(client, "get_context", {"query": "beta value"})
            assert "900" not in r1["context"]
            ref = await _tool(client, "report_refusal", {"read_id": r1["read_id"]})
            assert ref["payload"] is not None
            assert "[source excerpt] " + _G4_MISSED in ref["payload"]
            suc = await _tool(client, "report_success", {"read_id": r1["read_id"]})
            assert suc["confirmed"] is True          # event observed on a FACTORY cache
            fired = await _tool(client, "get_context", {"query": "value beta"})
            assert "900" in fired["context"]

    asyncio.run(flow())


@needs_mcp
def test_factory_persistence_restart(tmp_path: Path) -> None:
    # F5: the G5 bar with the STORE constructed inside the factory (their knob).
    db = str(tmp_path / "byo.db")
    ret = _DictRetriever()
    ret.set("kb:notes", _G4_DOC)
    store1 = SQLiteCognitionStore(db)
    synth1 = _DropNumberSynth()
    app1 = CoalentMCP(cache=_factory_cache(ret, synth1, store=store1))

    async def session_one() -> None:
        async with Client(app1.server) as client:
            await _tool(client, "get_context", {"query": "alpha beta value"})
            r1 = await _tool(client, "get_context", {"query": "beta value"})
            assert (await _tool(client, "report_refusal",
                                {"read_id": r1["read_id"]}))["payload"] is not None
            assert (await _tool(client, "report_success",
                                {"read_id": r1["read_id"]}))["confirmed"] is True

    asyncio.run(session_one())
    store1.close()                                   # kill the server

    synth2 = _DropNumberSynth()
    app2 = CoalentMCP(cache=_factory_cache(ret, synth2, store=SQLiteCognitionStore(db)))

    async def session_two() -> None:
        async with Client(app2.server) as client:
            stats = await _tool(client, "cache_stats")
            assert stats["units"] == 1               # loaded from the factory's store
            warm = await _tool(client, "get_context", {"query": "alpha beta value"})
            assert warm["cache_hit"] is True and synth2.calls == 0
            fired = await _tool(client, "get_context", {"query": "value beta"})
            assert "900" in fired["context"]         # confirmed key survived serde

    asyncio.run(session_two())


@needs_mcp
def test_factory_hybrid_watch_feeds_source_changed(tmp_path: Path) -> None:
    # Factory + --watch: the scan-only corpus fires source_changed on edits; the BYO
    # retriever stays THE substrate (ids must equal watch-relative paths — the
    # documented alignment requirement, exercised here).
    docs = tmp_path / "docs"
    _write(docs / "a.md", "Alpha value reached 42 units in June.")
    ret = _DictRetriever()
    ret.set("a.md", "Alpha value reached 42 units in June.")
    synth = _SentClaimSynth()
    app = CoalentMCP(docs, cache=_factory_cache(ret, synth))

    async def flow() -> None:
        async with Client(app.server) as client:
            cold = await _tool(client, "get_context", {"query": "alpha value"})
            assert "42" in cold["context"]
            listed = await _tool(client, "list_sources")   # watching IS active here
            assert [s["path"] for s in listed["sources"]] == ["a.md"]
            # The user's pipeline updates their index; the folder edit alone then
            # triggers invalidation on the very next read — no manual signal needed.
            _write(docs / "a.md", "Alpha value reached 99 units in June, revised.")
            ret.set("a.md", "Alpha value reached 99 units in June, revised.")
            nxt = await _tool(client, "get_context", {"query": "alpha value"})
            assert "99" in nxt["context"] and "42" not in nxt["context"]
            stats = await _tool(client, "cache_stats")
            assert stats["staleness_prevented"] == 1

    asyncio.run(flow())


@needs_mcp
def test_source_changed_manual_override_folder_mode(tmp_path: Path) -> None:
    # In folder mode the tool works as a manual override, but the SCAN stays
    # authoritative: the rebuild re-reads the folder's content, so a signal can
    # force a refresh yet can never make the cache diverge from disk.
    docs = tmp_path / "docs"
    _write(docs / "a.md", "Alpha value reached 42 units in June.")
    app = _make_app(docs, tmp_path / "state.db", _SentClaimSynth())

    async def flow() -> None:
        async with Client(app.server) as client:
            await _tool(client, "get_context", {"query": "alpha value"})
            sc = await _tool(client, "source_changed", {"artifact_id": "a.md"})
            assert sc["matched_units"] == 1 and len(sc["dirtied"]) == 1
            nxt = await _tool(client, "get_context", {"query": "alpha value"})
            assert nxt["cache_hit"] is False         # the override forced a rebuild...
            assert "42" in nxt["context"]            # ...from DISK content, unchanged
            stats = await _tool(client, "cache_stats")
            assert stats["staleness_prevented"] == 1

    asyncio.run(flow())


# ------------------------------------------------- v1.1 http transport + auth guard

def _start_http(app: CoalentMCP) -> tuple[Any, threading.Thread, str]:
    """Run the app's streamable-HTTP ASGI stack on an ephemeral localhost port in a
    daemon thread; return (uvicorn server, thread, url). Real sockets, loopback only —
    offline-deterministic."""
    import uvicorn

    config = uvicorn.Config(app.http_app(), host="127.0.0.1", port=0,
                            log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not server.started:
        assert time.time() < deadline, "http server did not start in 20s"
        time.sleep(0.02)
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, thread, f"http://127.0.0.1:{port}/mcp"


@needs_mcp
def test_http_two_concurrent_clients_share_one_cache(tmp_path: Path) -> None:
    ret = _DictRetriever()
    ret.set("kb:alpha", "Alpha value reached 42 units in June.")
    synth = _SentClaimSynth()
    app = CoalentMCP(cache=_factory_cache(ret, synth))
    server, thread, url = _start_http(app)
    try:
        async def flow() -> None:
            async def one_read(query: str) -> dict[str, Any]:
                async with Client(url) as client:
                    return await _tool(client, "get_context", {"query": query})

            # Two clients race the SAME query: the tool-body lock serializes them, so
            # exactly one build happens and the other is a shared warm hit — order-
            # independent, no corruption.
            r1, r2 = await asyncio.gather(one_read("alpha value"),
                                          one_read("alpha value"))
            assert sorted([r1["cache_hit"], r2["cache_hit"]]) == [False, True]
            assert "42" in r1["context"] and "42" in r2["context"]
            async with Client(url) as third:
                warm = await _tool(third, "get_context", {"query": "alpha value"})
                assert warm["cache_hit"] is True     # compounding is SHARED across clients
                stats = await _tool(third, "cache_stats")
                assert stats["units"] == 1 and stats["reads"] == 3

        asyncio.run(flow())
        assert synth.calls == 1                      # one build total, ever
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@needs_mcp
def test_http_bearer_auth_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx2

    from mcp.client.streamable_http import streamable_http_client

    monkeypatch.setenv("COALENT_MCP_TOKEN", "sekrit-token")   # read at app build time
    ret = _DictRetriever()
    ret.set("kb:alpha", "Alpha value reached 42 units in June.")
    app = CoalentMCP(cache=_factory_cache(ret, _SentClaimSynth()))
    server, thread, url = _start_http(app)
    try:
        # No token / wrong token: 401 before the MCP app sees a byte.
        assert httpx2.post(url, content=b"{}").status_code == 401
        assert httpx2.post(
            url, content=b"{}", headers={"Authorization": "Bearer wrong"}
        ).status_code == 401

        async def authed() -> None:
            async with httpx2.AsyncClient(
                headers={"Authorization": "Bearer sekrit-token"},
                timeout=httpx2.Timeout(10.0, read=60.0), follow_redirects=True,
            ) as http_client:
                transport = streamable_http_client(url, http_client=http_client)
                async with Client(transport) as client:
                    tools = (await client.list_tools()).tools
                    assert len(tools) == 7           # full surface behind the guard

        asyncio.run(authed())
    finally:
        server.should_exit = True
        thread.join(timeout=10)


# ------------------------------------------------------------------- G6 fail-loud
# Zero-dep on purpose: main() must refuse BEFORE it ever needs the SDK or OpenAI.

def test_g6_missing_api_key_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    docs = tmp_path / "docs"
    docs.mkdir()
    code = main(["--watch", str(docs)])
    assert code != 0
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY" in err and "never degrades" in err


def test_g6_bad_watch_dir_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--watch", str(tmp_path / "missing")])
    assert code != 0
    assert "not a directory" in capsys.readouterr().err


def test_g6_bad_budget_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")   # so the budget check is what fires
    docs = tmp_path / "docs"
    docs.mkdir()
    code = main(["--watch", str(docs), "--budget", "0"])
    assert code != 0
    assert "--budget" in capsys.readouterr().err


# ----------------------------------------------------- zero-dep units: corpus + glue

def test_watched_corpus_scan_lifecycle(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    _write(root / "a.md", "Alpha value reached 42 units in June.")
    _write(root / ".hidden" / "skip.md", "junk")     # hidden dirs are never watched
    _write(root / "notes.txt", "Beta kappa memo body here.")
    _write(root / "image.png", "binary-ish")         # unwatched extension
    corpus = WatchedCorpus(
        root, embedder=FunctionEmbedder(_embed), table_path=tmp_path / "scan.json")
    s0 = corpus.scan()
    assert sorted(s0.added) == ["a.md", "notes.txt"] and not s0.changed and not s0.removed
    hits = corpus.retrieve("alpha value")
    assert hits and hits[0].artifact_id == "a.md"
    assert hits[0].version                           # version = the file content hash

    # A TOUCH (identical content, newer mtime) is NOT a change — no spurious events.
    _write(root / "a.md", "Alpha value reached 42 units in June.")
    stat = (root / "a.md").stat()
    os.utime(root / "a.md", ns=(stat.st_atime_ns + 10**9, stat.st_mtime_ns + 10**9))
    s1 = corpus.scan()
    assert not s1.added and not s1.changed and not s1.removed

    # A real edit is a change; a new file is an add; a delete is a remove.
    _write(root / "a.md", "Alpha value reached 99 units in June, longer now.")
    _write(root / "b.md", "Gamma value data arrived.")
    (root / "notes.txt").unlink()
    s2 = corpus.scan()
    assert s2.changed == ["a.md"] and s2.added == ["b.md"] and s2.removed == ["notes.txt"]
    assert "99" in (corpus.text("a.md") or "")

    # peek() reports pending states without mutating.
    _write(root / "b.md", "Gamma value data arrived, revised and extended.")
    (root / "a.md").unlink()
    states = {s["path"]: s["state"] for s in corpus.peek()}
    assert states == {"a.md": "removed-pending", "b.md": "stale-pending"}
    assert corpus.paths() == ["a.md", "b.md"]        # peek changed nothing

    # Restart on the persisted scan table: an untouched folder yields ZERO events.
    s3 = corpus.scan()                               # settle the pending edits first
    assert s3.changed == ["b.md"] and s3.removed == ["a.md"]
    reborn = WatchedCorpus(
        root, embedder=FunctionEmbedder(_embed), table_path=tmp_path / "scan.json")
    s4 = reborn.scan()
    assert not s4.added and not s4.changed and not s4.removed
    assert reborn.retrieve("gamma value")            # …but the index is rebuilt and live


def test_chunk_paragraphs_merge_and_split() -> None:
    # Small paragraphs merge into one chunk; nothing but whitespace is lost.
    text = "First paragraph here.\n\nSecond one.\n\nThird one."
    assert chunk_paragraphs(text) == ["First paragraph here.\n\nSecond one.\n\nThird one."]
    # Paragraphs beyond the merge target split at paragraph boundaries.
    big = "\n\n".join(f"Paragraph {i} " + ("word " * 120) for i in range(4))
    chunks = chunk_paragraphs(big)
    assert len(chunks) == 4
    assert all(len(c) <= 1200 for c in chunks)
    # A single monster paragraph splits at whitespace under the hard cap.
    monster = "word " * 2000
    pieces = chunk_paragraphs(monster)
    assert all(len(p) <= 4000 for p in pieces)
    assert " ".join(pieces).split() == monster.split()
    assert chunk_paragraphs("   \n\n   ") == []


def test_has_source_reflects_provenance() -> None:
    retriever = InMemoryRetriever()
    retriever.add("doc:x", "alpha beta facts live here")
    cache = SemanticCache(retriever, StubSynthesizer())
    assert cache.has_source("doc:x") is False        # nothing read yet
    cache.get("alpha beta facts")
    assert cache.has_source("doc:x") is True
    assert cache.has_source("doc:y") is False


# --------------------------------------------- v1.1 zero-dep: factory loader + CLI

def test_load_cache_factory_resolves_and_fails_loud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)                      # cwd goes on sys.path (documented)
    _write(tmp_path / "factory_ok_mod.py", (
        "from coalent import InMemoryRetriever, SemanticCache, StubSynthesizer\n"
        "def build():\n"
        "    return SemanticCache(InMemoryRetriever(), StubSynthesizer())\n"
    ))
    _write(tmp_path / "factory_bare_mod.py", "x = 1\n")
    cache = load_cache_factory("factory_ok_mod:build")()
    assert isinstance(cache, SemanticCache)
    with pytest.raises(ValueError, match="module:function"):
        load_cache_factory("no_colon_here")
    with pytest.raises(ValueError, match="module:function"):
        load_cache_factory("mod:")
    with pytest.raises(ImportError, match="definitely_missing_mod"):
        load_cache_factory("definitely_missing_mod:build")
    with pytest.raises(AttributeError, match="build"):
        load_cache_factory("factory_bare_mod:build")


def test_main_requires_watch_or_factory(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) != 0
    assert "--watch" in capsys.readouterr().err


def test_main_factory_rejects_folder_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Silently-dead flags are refused loudly (the fail-loud principle, CLI edition).
    assert main(["--cache-factory", "m:f", "--store", "s.db"]) != 0
    assert "--store" in capsys.readouterr().err
    assert main(["--cache-factory", "m:f", "--budget", "500"]) != 0
    assert "--budget" in capsys.readouterr().err
    assert main(["--cache-factory", "m:f", "--ext", "md"]) != 0
    assert "--ext" in capsys.readouterr().err


def test_main_factory_import_fails_loud_and_needs_no_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # BYO mode demands no OPENAI_API_KEY (their models are theirs): with the key
    # deleted, the failure is the FACTORY's, never the key check.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code = main(["--cache-factory", "definitely_missing_mod:build"])
    assert code != 0
    err = capsys.readouterr().err
    assert "definitely_missing_mod" in err
    assert "OPENAI_API_KEY" not in err


@needs_mcp
def test_main_factory_wrong_return_type_fails_loud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _write(tmp_path / "factory_wrongtype_mod.py",
           "def build():\n    return 42\n")
    code = main(["--cache-factory", "factory_wrongtype_mod:build"])
    assert code != 0
    assert "SemanticCache" in capsys.readouterr().err


def test_scan_only_corpus_detects_but_never_retrieves(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    _write(root / "a.md", "Alpha value reached 42 units in June.")
    with pytest.raises(ValueError, match="embedder"):
        WatchedCorpus(root)                          # index mode needs an embedder
    corpus = WatchedCorpus(root, index=False)        # scan-only: no embedder at all
    s0 = corpus.scan()
    assert s0.added == ["a.md"]
    assert corpus.text("a.md") is not None
    with pytest.raises(RuntimeError, match="scan-only"):
        corpus.retrieve("alpha")                     # never masquerades as a retriever
    _write(root / "a.md", "Alpha value reached 99 units in June, revised.")
    s1 = corpus.scan()
    assert s1.changed == ["a.md"]
    assert "99" in (corpus.text("a.md") or "")
