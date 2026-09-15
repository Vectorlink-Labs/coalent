"""coalent.mcp — the Coalent MCP server: live context for any MCP client.

Two deployment modes, one tool surface:

**PRIMARY — bring your own vector DB (``--cache-factory``, v1.1).** Your factory
function returns a fully user-constructed :class:`SemanticCache` — your retriever /
vector DB, your embedder, your LLM, every knob (including ``store=`` for persistence
and ``pool_header=`` for attribution)::

    # my_cache.py — importable from the directory you launch in (cwd is on sys.path)
    def build() -> SemanticCache:
        return SemanticCache(my_qdrant_retriever, my_synth, embedder=my_embedder,
                             read_path="pool", residual_spans=True, query_keys=True,
                             pool_header=my_header, store=SQLiteCognitionStore("kb.db"))

    coalent-mcp --cache-factory my_cache:build [--transport http --port 8765]

Freshness in BYO mode is signal-driven: your ingestion pipeline calls the
``source_changed(artifact_id, text?)`` tool when a document changes and the affected
units invalidate immediately. Add ``--watch DIR`` alongside the factory and the folder
scan ALSO fires ``source_changed`` for edited files — honest caveat: watching never
ingests into YOUR index (two corpora, yours is the retriever); it only invalidates, and
only when your ``artifact_id``s equal the watch-relative file paths.

**Zero-config wedge — folder mode (``--watch``, v1).** Mounts the recommended v0.6
deployment (``read_path="pool"``, residual spans, query keys) over a directory of
documents: watched files are paragraph-chunked into the library's built-in vector
retriever, and every ``get_context`` call first does a cheap mtime/size scan — a file
whose content hash actually changed fires ``source_changed`` (new files ingest, deleted
files invalidate) BEFORE the read serves. You cannot get a stale answer after saving a
file. Attribution ships the golden path automatically: every ingested chunk carries
v0.7 metadata (title = first H1 else filename, source = relpath, date = mtime), served
as ``[{title} | {path} | {YYYY-MM-DD}]`` per source group — the measured 0.68-vs-0.73
header-ladder gap (n=605 graded; see cache.py) never opens. Folder mode requires
``OPENAI_API_KEY`` (embedder + gpt-4o-mini build synthesis) and fails LOUD without it —
never degrading to the lexical embedder (the pool-path guard's principle). Factory mode
needs no key: your factory brings its own models.

**Transports.** stdio (default; client-launched) or ``--transport http`` (the SDK's
streamable-HTTP): one long-lived process, many concurrent agents, ONE shared cache —
shared compounding, no store races. The library is not documented thread-safe, so a
single lock serializes every tool body (serves are ms-fast; the lock also means two
concurrent identical misses build once, not twice). Optional ``COALENT_MCP_TOKEN`` env
requires ``Authorization: Bearer`` on every http request — bind localhost or trusted
networks; real multi-tenant auth is the future cloud tier, deliberately not this.

Install: ``pip install "coalent[mcp]"`` (add ``,openai`` for folder mode). The ``mcp``
SDK import is guarded and lazy: this module imports cleanly without the extra; only
building/running the server requires it (and errors with the install command).

Out of scope for v1.1 (documented, not forgotten): MCP sampling (using the client's
LLM for builds), resources/prompts surfaces, multi-dir namespaces, non-text files.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib
import json
import os
import re
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, cast

from . import __version__
from .adapters.providers import OpenAIProvider
from .semantic.cache import Result, SemanticCache
from .semantic.embedding import Embedder, embed_texts
from .semantic.memory import InMemoryRetriever
from .semantic.ports import Chunk, Synthesizer
from .semantic.store import SQLiteCognitionStore
from .semantic.synthesizer import LLMSynthesizer
from .semantic.unit import Cognition

#: Default watched extensions (the ``--ext`` flag overrides; compared case-insensitively).
DEFAULT_EXTS: tuple[str, ...] = ("md", "txt", "mdx", "rst")
#: Default serve budget — the pool path's measured ~1k-token operating point (cache.py).
DEFAULT_BUDGET = 1000

# Paragraph chunking constants. Target ≈300 tokens/chunk under the library's 4-chars/token
# estimate — small enough that per-chunk provenance stays useful, big enough that a fact and
# its qualifier usually travel together. A single paragraph over the hard cap is split at
# whitespace so one pathological blob can't become a mega-chunk.
_CHUNK_TARGET_CHARS = 1200
_CHUNK_HARD_CAP_CHARS = 4000
_PARA_SPLIT = re.compile(r"\n\s*\n")
# The first markdown H1 line ("# Title") — the watched file's title for ingest metadata.
_H1 = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)

_INSTALL_HINT = 'pip install "coalent[mcp,openai]"'


def _hash_text(text: str) -> str:
    """Stable short content hash (sha256[:16]) — same shape the domain layer uses."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def chunk_paragraphs(text: str) -> list[str]:
    """Paragraph-based chunking (the library ships no chunker, so this is the ONE here):
    split on blank lines, then greedy-merge consecutive paragraphs up to
    ``_CHUNK_TARGET_CHARS``; a lone paragraph beyond ``_CHUNK_HARD_CAP_CHARS`` splits at
    the last whitespace before the cap. Deterministic; drops nothing but whitespace."""
    paras = [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paras:
        candidate = f"{buf}\n\n{para}" if buf else para
        if buf and len(candidate) > _CHUNK_TARGET_CHARS:
            chunks.append(buf)
            candidate = para
        buf = candidate
        while len(buf) > _CHUNK_HARD_CAP_CHARS:
            cut = buf.rfind(" ", 0, _CHUNK_HARD_CAP_CHARS)
            if cut <= 0:                      # no whitespace to split at: hard cut
                cut = _CHUNK_HARD_CAP_CHARS
            chunks.append(buf[:cut].rstrip())
            buf = buf[cut:].lstrip()
    if buf:
        chunks.append(buf)
    return chunks


class _CachingEmbedder:
    """Content-keyed memo over the real embedder for CORPUS chunks only, so a rescan
    re-embeds nothing but changed text (an unchanged 50-file folder re-ingests for free).
    Query embeddings pass straight through un-memoized — bounded memory by construction:
    the memo holds exactly the live corpus (``warm`` prunes to it)."""

    def __init__(self, inner: Embedder) -> None:
        self._inner = inner
        self._memo: dict[str, list[float]] = {}

    def warm(self, texts: list[str]) -> None:
        """Prune the memo to the live chunk set and batch-embed the misses in ONE
        provider round-trip (via ``embed_texts`` — the library's batching seam)."""
        live = set(texts)
        self._memo = {t: v for t, v in self._memo.items() if t in live}
        missing = [t for t in dict.fromkeys(texts) if t not in self._memo]
        for text, vec in zip(missing, embed_texts(self._inner, missing)):
            self._memo[text] = [float(x) for x in vec]

    def embed(self, text: str) -> list[float]:
        hit = self._memo.get(text)
        return list(hit) if hit is not None else list(self._inner.embed(text))

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        misses = [t for t in texts if t not in self._memo]
        if misses:  # batch the misses; memoization itself stays warm()-owned (bounded)
            fresh = dict(zip(misses, embed_texts(self._inner, misses)))
            return [list(self._memo.get(t) or fresh[t]) for t in texts]
        return [list(self._memo[t]) for t in texts]


@dataclass(slots=True)
class SourceState:
    """One watched file's last-ingested identity — what the rescan compares against."""

    mtime_ns: int
    size: int
    content_hash: str       # sha256[:16] of the newline-normalized text
    mtime: float            # for the "[path | modified YYYY-MM-DD]" header + list_sources


@dataclass(slots=True)
class ScanResult:
    """What one rescan found. ``changed`` means the CONTENT hash changed — a bare mtime
    touch (editor save with identical bytes, restart re-read) is deliberately not a
    change, so it can never trigger a spurious invalidation."""

    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)


class WatchedCorpus:
    """The watched-dir corpus: files → paragraph chunks → the library's built-in
    ``InMemoryRetriever`` (the shipped in-process vector store — cosine top-k over
    chunk embeddings; the same Retriever protocol any external vector DB would fill).

    In folder mode this object IS the cache's retriever (``retrieve`` delegates to the
    current inner index, which is rebuilt atomically on change — the cache's reference
    never dangles). With ``index=False`` (v1.1: the factory-hybrid mode) it is a pure
    CHANGE DETECTOR: it scans, hashes, and holds text so edits can fire
    ``source_changed``, but builds no index, needs no embedder, and refuses to
    retrieve — the factory cache's own retriever is the substrate.

    ``artifact_id`` = the file's watch-relative POSIX path: meaningful ids by
    construction, so provenance lines and invalidation events read as filenames.

    The scan table (path → mtime/size/hash) persists as JSON next to the store, so a
    restart re-ingests the folder WITHOUT firing invalidation for files that did not
    actually change while the server was down — provenance span hashes are chunk-level,
    so "changed" decisions must come from this table, never from hash-shape mismatches.
    Hidden paths (any ``.``-prefixed part — including the ``.coalent`` store dir) are
    never watched."""

    def __init__(
        self,
        root: str | Path,
        *,
        exts: tuple[str, ...] = DEFAULT_EXTS,
        embedder: Embedder | None = None,
        top_k: int = 6,
        table_path: str | Path | None = None,
        index: bool = True,
    ) -> None:
        self._root = Path(root).resolve()
        if not self._root.is_dir():
            raise NotADirectoryError(f"watch dir does not exist: {self._root}")
        self._exts = {e.lower().lstrip(".") for e in exts if e.strip()}
        if not self._exts:
            raise ValueError("at least one watched extension is required")
        self._index = index
        if index and embedder is None:
            raise ValueError("an embedder is required unless index=False (scan-only mode)")
        self._embedder = _CachingEmbedder(embedder) if (index and embedder) else None
        self._top_k = top_k
        self._table_path = Path(table_path) if table_path is not None else None
        self._files: dict[str, SourceState] = {}
        self._texts: dict[str, str] = {}
        self._retriever: InMemoryRetriever | None = (
            InMemoryRetriever(embedder=self._embedder, top_k=top_k)
            if self._embedder is not None else None
        )
        self._load_table()

    # ------------------------------------------------------------ Retriever protocol
    def retrieve(self, query: str, *, namespace: str | None = None) -> list[Chunk]:
        if self._retriever is None:
            # Fail loud: a scan-only corpus must never masquerade as a retriever — in
            # hybrid mode the factory cache's OWN retriever is the substrate.
            raise RuntimeError("this WatchedCorpus is scan-only (index=False); it "
                               "detects changes but does not retrieve")
        return self._retriever.retrieve(query, namespace=namespace)

    # ------------------------------------------------------------------- inspection
    def paths(self) -> list[str]:
        """The currently-ingested watch-relative paths, sorted."""
        return sorted(self._files)

    def text(self, path: str) -> str | None:
        """The last-ingested text of one watched file (None if unknown)."""
        return self._texts.get(path)

    def modified_date(self, path: str) -> str | None:
        """The file's last-ingested mtime as a local YYYY-MM-DD (header material)."""
        state = self._files.get(path)
        return datetime.fromtimestamp(state.mtime).strftime("%Y-%m-%d") if state else None

    def file_meta(self, path: str) -> dict[str, str]:
        """One watched file's v0.7 ingest metadata — the auto-wired ``Chunk.meta``:
        title = the first markdown H1 (else the filename), source = the watch-relative
        path (the artifact id itself), date = the last-ingested mtime (YYYY-MM-DD).
        Files HAVE metadata, so folder mode ships the measured metadata header rung
        (``[{title} | {source} | {date}]``) by default — never the bare fallbacks."""
        match = _H1.search(self._texts.get(path, ""))
        meta = {
            "title": match.group(1) if match else path.rsplit("/", 1)[-1],
            "source": path,
        }
        date = self.modified_date(path)
        if date:
            meta["date"] = date
        return meta

    def peek(self) -> list[dict[str, Any]]:
        """Freshness states WITHOUT mutating (stat-compare only, no reads, no events):
        ``fresh`` / ``stale-pending`` (stat differs; resolved on the next scan — a bare
        touch clears back to fresh then) / ``new-pending`` / ``removed-pending``."""
        on_disk = self._walk()
        out: list[dict[str, Any]] = []
        for rel in sorted(set(on_disk) | set(self._files)):
            prev, path = self._files.get(rel), on_disk.get(rel)
            if prev is None and path is not None:
                state, mtime = "new-pending", path.stat().st_mtime
            elif path is None and prev is not None:
                state, mtime = "removed-pending", prev.mtime
            else:
                assert prev is not None and path is not None
                st = path.stat()
                stale = st.st_mtime_ns != prev.mtime_ns or st.st_size != prev.size
                state, mtime = ("stale-pending" if stale else "fresh"), st.st_mtime
            out.append({
                "path": rel,
                "state": state,
                "last_modified": datetime.fromtimestamp(mtime).isoformat(timespec="seconds"),
            })
        return out

    # ------------------------------------------------------------------ the rescan
    def scan(self) -> ScanResult:
        """One full rescan: stat every watched file (the cheap path — an unchanged
        folder costs stats only), read + hash the ones whose stat moved, rebuild the
        retriever index when any text differs (index mode only), persist the scan
        table. Returns what actually changed BY CONTENT — the caller wires these into
        cache invalidation."""
        on_disk = self._walk()
        result = ScanResult()
        dirty_index = False
        for rel, path in on_disk.items():
            st = path.stat()
            prev = self._files.get(rel)
            if (prev is not None and rel in self._texts
                    and prev.mtime_ns == st.st_mtime_ns and prev.size == st.st_size):
                continue                      # stat unchanged and text held: nothing to do
            text = self._read(path)
            digest = _hash_text(text)
            if prev is None:
                result.added.append(rel)
            elif prev.content_hash != digest:
                result.changed.append(rel)
            # else: a touch (or restart re-read) — content identical, no event.
            if self._texts.get(rel) != text:
                dirty_index = True
            self._texts[rel] = text
            self._files[rel] = SourceState(
                mtime_ns=st.st_mtime_ns, size=st.st_size,
                content_hash=digest, mtime=st.st_mtime,
            )
        for rel in [r for r in self._files if r not in on_disk]:
            result.removed.append(rel)
            del self._files[rel]
            self._texts.pop(rel, None)
            dirty_index = True
        if dirty_index and self._index:
            self._rebuild_index()
        if result.added or result.changed or result.removed or dirty_index:
            self._save_table()
        return result

    def _walk(self) -> dict[str, Path]:
        """Watched files on disk right now: {relative POSIX path: absolute path},
        sorted for determinism; extension-filtered; hidden parts skipped."""
        found: dict[str, Path] = {}
        for path in sorted(self._root.rglob("*")):
            if not path.is_file() or path.suffix.lower().lstrip(".") not in self._exts:
                continue
            rel = path.relative_to(self._root)
            if any(part.startswith(".") for part in rel.parts):
                continue                      # hidden dirs/files, incl. .coalent itself
            found[rel.as_posix()] = path
        return found

    @staticmethod
    def _read(path: Path) -> str:
        """Read + newline-normalize (CRLF → LF) so content hashes are EOL-stable across
        editors and platforms. Undecodable bytes are replaced, never fatal — one odd
        file must not take down every read."""
        return path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")

    def _rebuild_index(self) -> None:
        """Swap in a fresh ``InMemoryRetriever`` over the current chunk set. The embed
        memo is batch-warmed first, so only genuinely new/changed chunk text costs an
        embedding call; the swap is atomic (build fully, then assign)."""
        assert self._embedder is not None     # only reachable in index mode
        rows: list[tuple[str, str, str, dict[str, str]]] = []
        for rel in sorted(self._texts):
            version = self._files[rel].content_hash
            meta = self.file_meta(rel)        # one dict per file, shared by its chunks
            for chunk in chunk_paragraphs(self._texts[rel]):
                rows.append((rel, chunk, version, meta))
        self._embedder.warm([text for _, text, _, _ in rows])
        retriever = InMemoryRetriever(embedder=self._embedder, top_k=self._top_k)
        for rel, text, version, meta in rows:
            retriever.add(rel, text, version=version, meta=meta)
        self._retriever = retriever

    # --------------------------------------------------------------- scan-table serde
    def _load_table(self) -> None:
        if self._table_path is None or not self._table_path.exists():
            return
        try:
            raw = json.loads(self._table_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return  # a corrupt table only costs a conservative full re-ingest
        for rel, row in raw.items():
            try:
                self._files[str(rel)] = SourceState(
                    mtime_ns=int(row["mtime_ns"]), size=int(row["size"]),
                    content_hash=str(row["content_hash"]), mtime=float(row["mtime"]),
                )
            except (KeyError, TypeError, ValueError):
                continue  # one poisoned row must not brick startup

    def _save_table(self) -> None:
        if self._table_path is None:
            return
        payload = {
            rel: {"mtime_ns": s.mtime_ns, "size": s.size,
                  "content_hash": s.content_hash, "mtime": s.mtime}
            for rel, s in self._files.items()
        }
        self._table_path.parent.mkdir(parents=True, exist_ok=True)
        self._table_path.write_text(json.dumps(payload), encoding="utf-8")


def _require_mcp() -> Any:
    """Import the MCP SDK server class, or fail with the install command. Guarded and
    lazy on purpose (zero-new-deps-in-core): everything above this line works without
    the extra; only building/running the server needs it."""
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            f"coalent.mcp needs the 'mcp' extra (the MCP python SDK): {_INSTALL_HINT}"
        ) from exc
    return MCPServer


def load_cache_factory(spec: str) -> Callable[[], SemanticCache]:
    """Resolve ``--cache-factory module:function`` to the callable — fail LOUD with the
    exact spec named on every failure shape (bad syntax, unimportable module, missing
    or non-callable attribute). The launch directory goes on ``sys.path`` first (the
    gunicorn/uvicorn app-loading convention), so ``my_cache:build`` works from the
    project root."""
    module_name, sep, func_name = spec.partition(":")
    if not sep or not module_name or not func_name:
        raise ValueError(f"--cache-factory must be 'module:function', got {spec!r}")
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"--cache-factory {spec!r}: cannot import module {module_name!r}: {exc}"
        ) from exc
    fn = getattr(module, func_name, None)
    if not callable(fn):
        raise AttributeError(
            f"--cache-factory {spec!r}: {module_name!r} has no callable {func_name!r}")
    # The return-type contract (a SemanticCache) is enforced at the call site by
    # CoalentMCP's isinstance guard — a cast, not a promise.
    return cast(Callable[[], SemanticCache], fn)


class _BearerAuthASGI:
    """Pure-ASGI bearer guard for the http transport (no starlette import — the wrapper
    stays dependency-neutral): every http request must carry ``Authorization: Bearer
    <COALENT_MCP_TOKEN>`` or gets a 401 before the MCP app sees a byte. Constant-time
    compare; lifespan (and any non-http) scopes pass through untouched. A shared secret
    for localhost/trusted networks — real multi-tenant auth is the future cloud tier,
    deliberately not this."""

    def __init__(self, app: Any, token: str) -> None:
        self._app = app
        self._expected = f"Bearer {token}"

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return
        supplied = ""
        for key, value in scope.get("headers", []):
            if key.decode("latin-1").lower() == "authorization":
                supplied = value.decode("latin-1")
                break
        if not hmac.compare_digest(supplied, self._expected):
            await send({"type": "http.response.start", "status": 401,
                        "headers": [(b"content-type", b"text/plain"),
                                    (b"www-authenticate", b"Bearer")]})
            await send({"type": "http.response.body", "body": b"unauthorized"})
            return
        await self._app(scope, receive, send)


_SERVER_INSTRUCTIONS = (
    "Coalent serves fresh, attributed facts from a provenance-invalidated cache. Facts "
    "are invalidated the instant their source changes: in folder-watch mode every "
    "get_context call rescans the watched files before serving; in BYO mode the "
    "ingestion pipeline signals edits via the source_changed tool. Workflow: call "
    "get_context(query) and use the returned context; if your answer attempt must "
    "refuse for lack of a fact, call report_refusal(read_id) and retry with the extra "
    "source excerpts it returns; if that retry succeeds, call report_success(read_id) "
    "so the cache learns the retrieval key permanently. list_sources / cache_stats / "
    "refresh are for inspection and forcing a rescan."
)


class CoalentMCP:
    """The wiring: a :class:`SemanticCache` + the 7-tool MCP surface, in two modes.

    **Factory (BYO) mode** — pass ``cache=``: the user's factory constructed the cache
    (their retriever/vector DB, embedder, LLM, every knob — including ``store=`` for
    persistence and ``pool_header=`` for attribution; the library itself warns at
    construction when the pool path lacks a header). The server adds ONLY protocol
    glue; ``budget``/``store_path``/``embedder``/``synthesizer`` are folder-mode
    arguments and are rejected loudly here. Freshness is signal-driven via the
    ``source_changed`` tool; pass ``watch=`` too and the folder scan ALSO fires it
    (scan-only corpus — never a second retriever; ids must equal watch-relative paths).

    **Folder mode** — pass ``watch=`` only: the shipped v0.6 deployment over a watched
    folder (``read_path="pool"``, residual spans, query keys, serve budget from
    ``budget``, auto ``pool_header`` from file metadata, SQLite serde persistence).
    ``embedder``/``synthesizer`` default to the OpenAI pair (text-embedding-3-small +
    gpt-4o-mini via ``LLMSynthesizer``'s extract-mode default); tests inject
    deterministic offline doubles.

    **Concurrency**: one ``threading.RLock`` serializes every tool body. The library is
    not documented thread-safe and the http transport serves many concurrent clients;
    serves are ms-fast so serialization is not the bottleneck, and it makes two
    concurrent identical misses build once instead of twice."""

    def __init__(
        self,
        watch: str | Path | None = None,
        *,
        cache: SemanticCache | None = None,
        exts: tuple[str, ...] = DEFAULT_EXTS,
        budget: int = DEFAULT_BUDGET,
        store_path: str | Path | None = None,
        embedder: Embedder | None = None,
        synthesizer: Synthesizer | None = None,
        top_k: int = 6,
    ) -> None:
        if cache is not None and not isinstance(cache, SemanticCache):
            raise TypeError(
                f"cache factory must return a SemanticCache, got {type(cache).__name__}")
        if cache is None and watch is None:
            raise ValueError(
                "either watch= (folder mode) or cache= (a factory-built SemanticCache) "
                "is required")
        self._lock = threading.RLock()
        self._event_counts: dict[str, int] = {}
        self.corpus: WatchedCorpus | None = None
        self.watch_dir: Path | None = None
        self.store_path: Path | None = None
        self.store: SQLiteCognitionStore | None = None
        if cache is not None:
            # ---------------- factory (BYO) mode: the user's cache, our protocol glue.
            # Folder-mode knobs are rejected LOUDLY — silently ignoring them would let
            # an operator believe --store/--budget took effect when the factory owns
            # persistence and serve_budget outright.
            if store_path is not None:
                raise ValueError(
                    "store_path/--store has no effect in factory mode — construct your "
                    "cache with store=SQLiteCognitionStore(...) inside the factory")
            if embedder is not None or synthesizer is not None:
                raise ValueError(
                    "embedder/synthesizer are folder-mode arguments — in factory mode "
                    "the factory constructs the cache with its own")
            self.cache = cache
            self._observe_events(cache)
            if watch is not None:
                # Hybrid: the folder scan feeds source_changed ALONGSIDE their index.
                # Scan-only (index=False): no embedder, no second retriever — watching
                # never ingests into their index; it only invalidates, and only when
                # their artifact_ids equal the watch-relative paths (documented caveat).
                watch_dir = Path(watch).resolve()
                if not watch_dir.is_dir():
                    raise NotADirectoryError(f"--watch {watch!s} is not a directory")
                self.watch_dir = watch_dir
                self.corpus = WatchedCorpus(
                    watch_dir, exts=exts, index=False,
                    table_path=watch_dir / ".coalent" / "scan.json",
                )
                self._apply(self.corpus.scan())
        else:
            # ---------------- folder mode: the shipped v0.6 deployment (v1, unchanged).
            assert watch is not None
            watch_dir = Path(watch).resolve()
            if not watch_dir.is_dir():
                raise NotADirectoryError(f"--watch {watch!s} is not a directory")
            if budget <= 0:
                raise ValueError(f"--budget must be positive, got {budget}")
            if embedder is None or synthesizer is None:
                # Fail LOUD, never degrade: without a key the library would fall back to
                # the lexical HashingEmbedder, which the pool read path rightly refuses.
                if not os.environ.get("OPENAI_API_KEY"):
                    raise RuntimeError(
                        "OPENAI_API_KEY is not set. coalent-mcp needs it for semantic "
                        "embeddings and build synthesis (gpt-4o-mini); it never degrades "
                        "to the lexical embedder. Set the key, or construct CoalentMCP "
                        "with explicit embedder=/synthesizer=."
                    )
            if embedder is None:
                from .semantic.embedding import OpenAIEmbedder  # lazy: the openai extra

                embedder = OpenAIEmbedder()
            if synthesizer is None:
                synthesizer = LLMSynthesizer(OpenAIProvider())  # gpt-4o-mini, extract=True
            self.watch_dir = watch_dir
            store = Path(store_path) if store_path is not None else (
                watch_dir / ".coalent" / "store.db")
            store.parent.mkdir(parents=True, exist_ok=True)
            self.store_path = store
            self.corpus = WatchedCorpus(
                watch_dir, exts=exts, embedder=embedder, top_k=top_k,
                table_path=Path(str(store) + ".scan.json"),
            )
            self.store = SQLiteCognitionStore(str(store))
            self.cache = SemanticCache(
                self.corpus,
                synthesizer,
                embedder=embedder,
                read_path="pool",
                serve_budget=budget,
                residual_spans=True,
                query_keys=True,
                pool_header=self._pool_header,
                store=self.store,
                on_event=self._on_event,
            )
            # Startup sync: ingest the folder and invalidate exactly what changed while
            # the server was down (the persisted scan table makes "changed" content-
            # exact, so an untouched folder restarts to a fully WARM cache — G5's bar).
            self._apply(self.corpus.scan())
        self.server: Any = self._build_server()

    def close(self) -> None:
        """Release the folder-mode SQLite store (a factory cache's store belongs to the
        factory; restart hygiene — the OS would get it anyway)."""
        if self.store is not None:
            self.store.close()

    # ------------------------------------------------------------------ cache glue
    def _on_event(self, event: dict[str, Any]) -> None:
        """Count the cache's structured events by kind — report_success reads the
        ``key_confirmed`` delta to answer honestly instead of guessing."""
        kind = str(event.get("event", ""))
        self._event_counts[kind] = self._event_counts.get(kind, 0) + 1

    def _observe_events(self, cache: SemanticCache) -> None:
        """Factory mode: observe the cache's events WITHOUT displacing the factory's
        own ``on_event`` hook — ours runs first, theirs still fires. An in-package
        private seam (``_on_event``), version-locked with the library like the budget
        override; the cache already swallows hook exceptions, so chaining adds no new
        failure mode."""
        user_hook = cache._on_event
        if user_hook is None:
            cache._on_event = self._on_event
            return

        def chained(event: dict[str, Any],
                    _user: Callable[[dict[str, Any]], None] = user_hook) -> None:
            self._on_event(event)
            _user(event)

        cache._on_event = chained

    def _pool_header(self, unit: Cognition) -> str:
        """Folder mode's automatic attribution header. Units built from watched files
        carry ``source_meta`` (v0.7 ingest metadata via ``WatchedCorpus.file_meta``:
        title = first H1 else filename, source = relpath, date = mtime), which the
        library's default ladder renders as ``[{title} | {path} | {YYYY-MM-DD}]``.
        Files HAVE metadata, so this deployment ships the measured golden path by
        default — the 0.68-vs-0.73 bare-header gap (cache.py's construction warning)
        cannot open here. A pre-v0.7 unit (persisted before meta existed) keeps the v1
        ``[{path} | modified {YYYY-MM-DD}]`` form from the scan table, then the bare
        path, then the library's default header shape. Factory mode never wires this:
        attribution there is the factory's ``pool_header=`` (the library warns at
        construction if absent)."""
        if unit.source_meta:
            return SemanticCache._default_header(unit)
        artifact = unit.evidence[0].artifact_id if unit.evidence else ""
        if artifact:
            date = self.corpus.modified_date(artifact) if self.corpus else None
            return f"[{artifact} | modified {date}]" if date else f"[{artifact}]"
        query = (unit.query or "").strip()
        return ("## " + query[:60]) if query else f"[source: {unit.id}]"

    def _apply(self, scan: ScanResult) -> dict[str, Any]:
        """Wire one scan's findings into the cache: changed content invalidates by
        provenance (only for files some unit actually read — ``has_source`` keeps the
        library's matched-no-units wiring warning honest), deletions evict. Additions
        need no event: no unit depends on a never-read file."""
        dirtied = 0
        evicted = 0
        for rel in scan.changed:
            if self.cache.has_source(rel) and self.corpus is not None:
                text = self.corpus.text(rel)
                outcome = self.cache.source_changed(rel, text=text)
                dirtied += len(outcome.dirtied)
                evicted += len(outcome.deleted)
        for rel in scan.removed:
            if self.cache.has_source(rel):
                evicted += len(self.cache.source_deleted(rel).deleted)
        return {
            "added": list(scan.added),
            "changed": list(scan.changed),
            "removed": list(scan.removed),
            "units_dirtied": dirtied,
            "units_evicted": evicted,
        }

    def _sync(self) -> dict[str, Any]:
        """The folder-scan freshness moment: rescan + invalidate before anything
        serves. A no-op shape in pure factory mode — there, freshness arrives through
        the ``source_changed`` tool, not a scan."""
        if self.corpus is None:
            return {"added": [], "changed": [], "removed": [],
                    "units_dirtied": 0, "units_evicted": 0}
        return self._apply(self.corpus.scan())

    def _sources_of(self, result: Result) -> list[str]:
        """The artifact ids behind a read, served order first: each served pool claim's
        owning unit contributes its evidence artifacts (via the public ``drill``), then
        any escalation raw evidence. De-duplicated, order-preserving."""
        out: list[str] = []
        for claim in result.pool:
            for chunk in self.cache.drill(claim.unit_id):
                if chunk.artifact_id and chunk.artifact_id not in out:
                    out.append(chunk.artifact_id)
        for chunk in result.evidence:
            if chunk.artifact_id and chunk.artifact_id not in out:
                out.append(chunk.artifact_id)
        return out

    # ------------------------------------------------------------------ tool bodies
    # Plain methods so the logic is directly testable; the MCP layer below is thin.
    # EVERY body takes the lock: the library is not documented thread-safe, and the
    # http transport runs tools from concurrent client sessions.
    def tool_get_context(self, query: str, budget: int | None = None) -> dict[str, Any]:
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if budget is not None and budget <= 0:
            raise ValueError(f"budget must be positive, got {budget}")
        with self._lock:
            self._sync()   # folder scan: never serve past a saved file — the guarantee
            # Per-call budget override: the library fixes serve_budget at construction,
            # so v1 swaps the private field around the read (in-package, version-locked
            # with the library — one PyPI artifact) and restores it under finally.
            previous = self.cache._serve_budget
            if budget is not None:
                self.cache._serve_budget = budget
            try:
                result = self.cache.get(query)
            finally:
                self.cache._serve_budget = previous
            parts: list[str] = []
            pool_text = str(result.context.get("pool", "") or "")
            if pool_text:
                parts.append(pool_text)
            elif result.context:
                # A factory cache MAY run the unit read path ("every knob" includes
                # read_path): its context is the projection dict, served as compact
                # JSON rather than invented prose. Raw stays separate below.
                unit_ctx = {k: v for k, v in result.context.items() if k != "raw"}
                if unit_ctx:
                    parts.append(json.dumps(unit_ctx, ensure_ascii=False))
            raw = result.context.get("raw")
            if isinstance(raw, list) and raw:   # the RAG floor fired: attributed raw
                parts.append("\n\n".join(str(r) for r in raw))
            return {
                "context": "\n\n".join(parts),
                "read_id": result.read_id,
                "sources": self._sources_of(result),
                "cache_hit": result.cache_hit,
                # v0.7 read surface (additive keys; the boundary port): the read's own
                # coverage + doubt, so an MCP-side evaluator can act without drilling.
                # gaps is non-empty only when the (factory-built) cache armed
                # gap_detector=True — folder mode never arms it.
                "coverage": round(result.coverage, 4),
                "needs_retrieval": result.needs_retrieval,
                "gaps": list(result.gaps),
            }

    def tool_report_refusal(self, read_id: str) -> dict[str, Any]:
        with self._lock:
            return {"payload": self.cache.report_refusal(read_id)}

    def tool_report_success(self, read_id: str) -> dict[str, Any]:
        with self._lock:
            before = self._event_counts.get("key_confirmed", 0)
            self.cache.report_success(read_id)
            return {"confirmed": self._event_counts.get("key_confirmed", 0) > before}

    def tool_source_changed(self, artifact_id: str, text: str | None = None) -> dict[str, Any]:
        """The BYO freshness signal (v1.1): the ingestion pipeline calls this when a
        document changes; units derived from it invalidate immediately (with ``text``,
        unchanged content is hash-detected and skipped). In folder mode it works as a
        manual override, but the SCAN stays authoritative: rebuilds re-read the watched
        folder's content, so a signal cannot make the cache diverge from disk."""
        if not artifact_id or not artifact_id.strip():
            raise ValueError("artifact_id must be a non-empty string")
        with self._lock:
            outcome = self.cache.source_changed(artifact_id, text=text)
            return {
                "matched_units": outcome.matched_units,
                "dirtied": list(outcome.dirtied),
                "skipped_unchanged": list(outcome.skipped_unchanged),
                "deleted": list(outcome.deleted),
            }

    def tool_list_sources(self) -> dict[str, Any]:
        with self._lock:
            return {
                "watch_dir": str(self.watch_dir) if self.watch_dir else None,
                "sources": self.corpus.peek() if self.corpus is not None else [],
            }

    def tool_cache_stats(self) -> dict[str, Any]:
        with self._lock:
            return dict(self.cache.stats())

    def tool_refresh(self) -> dict[str, Any]:
        with self._lock:
            return self._sync()

    # ------------------------------------------------------------------ MCP surface
    def _build_server(self) -> Any:
        """Register the seven tools on an ``MCPServer``. Tool descriptions are
        marketing surface (spec): each states the freshness/attribution guarantee
        plainly. Registration uses ``add_tool`` (not the decorator) so the functions
        keep their own types under mypy strict while the SDK object stays ``Any`` —
        the zero-dep CI leg typechecks this module without the extra installed."""
        mcp_server_cls = _require_mcp()
        server: Any = mcp_server_cls(
            name="coalent",
            version=__version__,
            instructions=_SERVER_INSTRUCTIONS,
        )

        def get_context(query: str, budget: int | None = None) -> dict[str, Any]:
            return self.tool_get_context(query, budget)

        def report_refusal(read_id: str) -> dict[str, Any]:
            return self.tool_report_refusal(read_id)

        def report_success(read_id: str) -> dict[str, Any]:
            return self.tool_report_success(read_id)

        def source_changed(artifact_id: str, text: str | None = None) -> dict[str, Any]:
            return self.tool_source_changed(artifact_id, text)

        def list_sources() -> dict[str, Any]:
            return self.tool_list_sources()

        def cache_stats() -> dict[str, Any]:
            return self.tool_cache_stats()

        def refresh() -> dict[str, Any]:
            return self.tool_refresh()

        server.add_tool(get_context, description=(
            "Fetch fresh, attributed context for a query. Facts are invalidated the "
            "instant their source changes: with folder watching active every call "
            "first rescans the watched files (mtime + content hash) and re-derives "
            "anything a save touched BEFORE serving; in BYO mode the ingestion "
            "pipeline signals edits via source_changed. Returns the budget-packed "
            "fact payload with per-source attribution headers, the contributing "
            "source ids, whether it was served from cache, a read_id for "
            "report_refusal / report_success, plus the read's coverage score, a "
            "needs_retrieval hint (the cache under-covered even after its own "
            "recovery), and any detected gaps (empty unless the cache was built "
            "with gap_detector=True)."
        ))
        server.add_tool(report_refusal, description=(
            "Your answerer refused over a get_context payload? Send that read_id. "
            "Returns a retry payload of up to 2 attributed verbatim source excerpts "
            "the extraction missed (payload=null if none qualifies) — append it to "
            "the original context and retry once. Costs zero LLM/retrieval calls."
        ))
        server.add_tool(report_success, description=(
            "Confirm that a retry over a report_refusal payload SUCCEEDED. The "
            "refused query becomes a durable alternate retrieval key on the fact "
            "that rescued it (persisted across restarts), so future paraphrases "
            "retrieve that fact directly. confirmed=false if the read_id attached "
            "no keys or already expired."
        ))
        server.add_tool(source_changed, description=(
            "Signal that a source document changed (the BYO freshness feed): every "
            "cached fact derived from artifact_id is invalidated immediately and "
            "re-derives on its next read. Pass the new text and unchanged content is "
            "hash-detected and skipped (no wasted rebuilds). In folder-watch mode "
            "this works as a manual override, but the file scan remains "
            "authoritative — rebuilds always re-read the watched folder's content."
        ))
        server.add_tool(list_sources, description=(
            "The watched files with freshness state: fresh (served facts match the "
            "file on disk), stale-pending (file changed on disk; its derived facts "
            "are invalidated on the next get_context/refresh), new-pending (not yet "
            "ingested), removed-pending (deleted; its facts evict on the next scan) "
            "— plus last-modified timestamps. Empty without folder watching (BYO "
            "sources live in your own index). Read-only: never mutates the cache."
        ))
        server.add_tool(cache_stats, description=(
            "Cache observability counters: units, reads, hits, hit_rate, "
            "staleness_prevented (reads that would have served stale knowledge and "
            "were rebuilt instead — the freshness guarantee, measured), invalidated "
            "units, token/cost telemetry, and the pool-path gate diagnostics."
        ))
        server.add_tool(refresh, description=(
            "Force a full rescan of the watched folder right now (get_context does "
            "this automatically before every read). Returns which files were added / "
            "changed / removed and how many cached units were invalidated or "
            "evicted. A no-op shape without folder watching — BYO freshness flows "
            "through source_changed instead."
        ))
        return server

    # -------------------------------------------------------------- http transport
    def http_app(self, *, host: str = "127.0.0.1") -> Any:
        """The streamable-HTTP ASGI app (SDK-built), optionally wrapped in the bearer
        guard when ``COALENT_MCP_TOKEN`` is set (read HERE, at app build time).
        ``host`` feeds the SDK's DNS-rebinding protection defaults for localhost."""
        app: Any = self.server.streamable_http_app(host=host)
        token = os.environ.get("COALENT_MCP_TOKEN")
        return _BearerAuthASGI(app, token) if token else app

    def run_http(self, *, host: str = "127.0.0.1", port: int = 8000) -> None:
        """Serve over streamable HTTP: ONE long-lived process, many concurrent MCP
        clients, one shared cache — shared compounding, no store races (the tool-body
        lock serializes cache access). uvicorn ships with the SDK, imported lazily so
        stdio deployments never touch it."""
        import uvicorn

        config = uvicorn.Config(
            self.http_app(host=host), host=host, port=port, log_level="info")
        uvicorn.Server(config).run()


# ---------------------------------------------------------------------- CLI entry
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coalent-mcp",
        description=(
            "Coalent MCP server: fresh, attributed facts from a provenance-invalidated "
            "cache — invalidated the moment a source changes. PRIMARY mode: "
            "--cache-factory module:function returning your own SemanticCache (your "
            "vector DB / embedder / LLM / every knob). Zero-config mode: --watch DIR "
            "over a folder of documents. stdio by default; --transport http for one "
            "long-lived server shared by many agents."
        ),
    )
    parser.add_argument(
        "--cache-factory", default=None, metavar="MODULE:FUNCTION",
        help="PRIMARY (BYO) mode: import MODULE and call FUNCTION() for a fully "
             "user-constructed SemanticCache; the folder flags below then no longer "
             "apply (--watch may still be added to feed source_changed on file edits)",
    )
    parser.add_argument(
        "--watch", default=None,
        help="directory of documents to watch (folder mode; optional change feed "
             "alongside --cache-factory)",
    )
    parser.add_argument(
        "--ext", default=None,
        help=f"comma-separated watched extensions (default: {','.join(DEFAULT_EXTS)}; "
             "folder watching only)",
    )
    parser.add_argument(
        "--budget", type=int, default=None,
        help=f"serve budget in estimated tokens (default: {DEFAULT_BUDGET}; folder "
             "mode only — a factory cache sets serve_budget itself)",
    )
    parser.add_argument(
        "--store", default=None,
        help="persistence path (default: <watch-dir>/.coalent/store.db; folder mode "
             "only — a factory cache passes store= itself)",
    )
    parser.add_argument(
        "--transport", choices=("stdio", "http"), default="stdio",
        help="stdio (default; client-launched) or http (streamable HTTP: one shared "
             "long-lived server for many agents; set COALENT_MCP_TOKEN for bearer "
             "auth)",
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="http bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000,
                        help="http bind port (default: 8000)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Startup is fail-loud by design (G6): a missing watch dir, missing key, broken
    factory, conflicting flags, or missing extra exits nonzero with the fix in one
    line — never a degraded or silently-misconfigured server."""
    args = build_arg_parser().parse_args(argv)
    factory_mode = args.cache_factory is not None
    if not factory_mode and args.watch is None:
        print("coalent-mcp: one of --watch (folder mode) or --cache-factory (BYO "
              "mode) is required", file=sys.stderr)
        return 2
    if factory_mode:
        # Folder-mode flags with a factory would be silently dead — refuse them loudly.
        if args.store is not None:
            print("coalent-mcp: --store has no effect with --cache-factory — construct "
                  "your cache with store=SQLiteCognitionStore(...) inside the factory",
                  file=sys.stderr)
            return 2
        if args.budget is not None:
            print("coalent-mcp: --budget has no effect with --cache-factory — the "
                  "factory cache sets serve_budget itself (per-call budget on "
                  "get_context still works)", file=sys.stderr)
            return 2
        if args.ext is not None and args.watch is None:
            print("coalent-mcp: --ext only shapes folder watching — add --watch or "
                  "drop --ext", file=sys.stderr)
            return 2
    if args.watch is not None and not Path(args.watch).is_dir():
        print(f"coalent-mcp: --watch {args.watch!r} is not a directory", file=sys.stderr)
        return 2
    if args.budget is not None and args.budget <= 0:
        print(f"coalent-mcp: --budget must be positive, got {args.budget}",
              file=sys.stderr)
        return 2
    if not factory_mode and not os.environ.get("OPENAI_API_KEY"):
        # Folder mode only: a factory brings its own embedder/LLM, so no key is
        # demanded there — the BYO principle, enforced.
        print(
            "coalent-mcp: OPENAI_API_KEY is not set. The server needs it for semantic "
            "embeddings (text-embedding-3-small) and build synthesis (gpt-4o-mini); it "
            "never degrades to the lexical embedder. Set the key and rerun:\n"
            "  export OPENAI_API_KEY=sk-...   (or setx on Windows)",
            file=sys.stderr,
        )
        return 2
    cache: SemanticCache | None = None
    if factory_mode:
        try:
            factory = load_cache_factory(args.cache_factory)
            cache = factory()
        except Exception as exc:  # noqa: BLE001 — startup surface: name the spec, exit loud
            print(f"coalent-mcp: --cache-factory {args.cache_factory!r} failed: {exc}",
                  file=sys.stderr)
            return 2
    try:
        _require_mcp()
    except ImportError as exc:
        print(f"coalent-mcp: {exc}", file=sys.stderr)
        return 2
    exts = (tuple(e.strip() for e in str(args.ext).split(",") if e.strip())
            if args.ext is not None else DEFAULT_EXTS)
    try:
        app = CoalentMCP(
            Path(args.watch) if args.watch else None,
            cache=cache,
            exts=exts,
            budget=args.budget if args.budget is not None else DEFAULT_BUDGET,
            store_path=Path(args.store) if args.store else None,
        )
    except ImportError as exc:  # the openai extra is missing — same fail-loud shape
        print(f"coalent-mcp: {exc} — {_INSTALL_HINT}", file=sys.stderr)
        return 2
    except (NotADirectoryError, RuntimeError, TypeError, ValueError) as exc:
        print(f"coalent-mcp: {exc}", file=sys.stderr)
        return 2
    if args.transport == "http":
        app.run_http(host=args.host, port=args.port)
    else:
        app.server.run("stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
