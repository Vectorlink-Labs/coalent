"""``coalent`` — a redis-CLI-style tool to inspect and manage a cognitive cache.

Operates on a persistent SQLite store (``--db``, or ``$COALENT_DB``, default
``coalent.db``) — the same file you pass to ``SemanticCache(store=...)``::

    coalent ls                              # what's cached
    coalent show cog:ab12cd34ef56           # understanding + provenance + raw
    coalent dirty cog:ab12cd34ef56          # mark one unit stale
    coalent invalidate confluence:98231     # dirty every unit that used a source
    coalent evict cog:ab12cd34ef56          # remove one unit
    coalent purge --yes                     # remove everything
    coalent stats                           # counts
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Callable

from .domain.models import Status
from .semantic.serde import cognition_to_dict
from .semantic.store import SQLiteCognitionStore

Command = Callable[[SQLiteCognitionStore, argparse.Namespace], int]


def _age(epoch: float) -> str:
    delta = max(0.0, time.time() - epoch)
    if delta < 90:
        return f"{int(delta)}s"
    if delta < 5400:
        return f"{int(delta / 60)}m"
    if delta < 129600:
        return f"{int(delta / 3600)}h"
    return f"{int(delta / 86400)}d"


def _preview(text: str, width: int = 50) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 3] + "..."


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def cmd_ls(store: SQLiteCognitionStore, args: argparse.Namespace) -> int:
    units = sorted(store.all(), key=lambda u: u.last_access, reverse=True)
    if not units:
        print("(cache is empty)")
        return 0
    print(f"{'STATUS':7} {'HITS':>4} {'AGE':>4} {'SRC':>3}  {'ID':18}  QUERY")
    for unit in units:
        print(
            f"{unit.status.value:7} {unit.hits:>4} {_age(unit.freshness_epoch):>4} "
            f"{len(unit.provenance.artifact_ids()):>3}  {unit.id:18}  {_preview(unit.query)}"
        )
    return 0


def cmd_show(store: SQLiteCognitionStore, args: argparse.Namespace) -> int:
    unit = store.get(args.id)
    if unit is None:
        print(f"not found: {args.id}")
        return 1
    if args.json:
        print(json.dumps(cognition_to_dict(unit), indent=2, ensure_ascii=False))
        return 0

    print(f"id:        {unit.id}")
    print(f"namespace: {unit.namespace or '(default)'}")
    print(f"query:     {unit.query}")
    print(
        f"status:    {unit.status.value}    hits: {unit.hits}    "
        f"age: {_age(unit.freshness_epoch)}"
    )
    print()
    print("understanding")
    print(_indent(json.dumps(unit.understanding, indent=2, ensure_ascii=False)))
    print()
    print(f"provenance ({len(unit.provenance.source_spans)} span(s))")
    for span in unit.provenance.source_spans:
        print(
            f"  - {span.artifact_id}  @{span.version or '-'}  "
            f"#{span.span or '*'}  {span.content_hash[:10] or '-'}"
        )
    print()
    print(f"raw evidence ({len(unit.evidence)} chunk(s))")
    for chunk in unit.evidence:
        print(f"  -- {chunk.artifact_id}  @{chunk.version or '-'}")
        body = chunk.text if args.full else _preview(chunk.text, 240)
        print(_indent(body, "     "))
    return 0


def cmd_dirty(store: SQLiteCognitionStore, args: argparse.Namespace) -> int:
    unit = store.get(args.id)
    if unit is None:
        print(f"not found: {args.id}")
        return 1
    unit.mark_dirty()
    store.put(unit)
    print(f"marked dirty: {args.id}")
    return 0


def cmd_invalidate(store: SQLiteCognitionStore, args: argparse.Namespace) -> int:
    dirtied: list[str] = []
    for unit in store.all():
        if args.artifact in unit.provenance.artifact_ids():
            unit.mark_dirty()
            store.put(unit)
            dirtied.append(unit.id)
    if dirtied:
        print(f"dirtied {len(dirtied)}: {', '.join(dirtied)}")
    else:
        print(f"no cached units used {args.artifact!r}")
    return 0


def cmd_evict(store: SQLiteCognitionStore, args: argparse.Namespace) -> int:
    if store.get(args.id) is None:
        print(f"not found: {args.id}")
        return 1
    store.delete(args.id)
    print(f"evicted: {args.id}")
    return 0


def cmd_purge(store: SQLiteCognitionStore, args: argparse.Namespace) -> int:
    ids = [unit.id for unit in store.all()]
    if not args.yes:
        print(f"refusing to purge {len(ids)} unit(s) — pass --yes to confirm")
        return 1
    for unit_id in ids:
        store.delete(unit_id)
    print(f"purged {len(ids)} unit(s)")
    return 0


def cmd_stats(store: SQLiteCognitionStore, args: argparse.Namespace) -> int:
    units = store.all()
    fresh = sum(1 for u in units if u.status == Status.FRESH)
    dirty = sum(1 for u in units if u.status == Status.DIRTY)
    artifacts: set[str] = set()
    for unit in units:
        artifacts |= set(unit.provenance.artifact_ids())
    print(f"units:             {len(units)}")
    print(f"  fresh:           {fresh}")
    print(f"  dirty:           {dirty}")
    print(f"tracked artifacts: {len(artifacts)}")
    return 0


_COMMANDS: dict[str, Command] = {
    "ls": cmd_ls,
    "show": cmd_show,
    "dirty": cmd_dirty,
    "invalidate": cmd_invalidate,
    "evict": cmd_evict,
    "purge": cmd_purge,
    "stats": cmd_stats,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coalent", description="Inspect and manage a Coalent cognitive cache."
    )
    parser.add_argument(
        "--db",
        default=os.environ.get("COALENT_DB", "coalent.db"),
        help="SQLite store path (default: coalent.db or $COALENT_DB)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ls", help="list cognition units")
    show = sub.add_parser("show", help="show a unit's understanding, provenance, and raw evidence")
    show.add_argument("id")
    show.add_argument("--full", action="store_true", help="print full raw evidence (not truncated)")
    show.add_argument("--json", action="store_true", help="emit the full unit as JSON")
    dirty = sub.add_parser("dirty", help="manually mark one unit stale")
    dirty.add_argument("id")
    inv = sub.add_parser("invalidate", help="dirty every unit that used a source artifact")
    inv.add_argument("artifact")
    evict = sub.add_parser("evict", help="remove one unit")
    evict.add_argument("id")
    purge = sub.add_parser("purge", help="remove all units")
    purge.add_argument("--yes", action="store_true", help="confirm the purge")
    sub.add_parser("stats", help="show cache counts")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = SQLiteCognitionStore(args.db)
    try:
        return _COMMANDS[args.command](store, args)
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
