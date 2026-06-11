"""Tests for the `coalent` CLI over a SQLite-backed cognition store."""
from __future__ import annotations

from coalent.cli import main
from coalent.semantic import InMemoryRetriever, SemanticCache, SQLiteCognitionStore, StubSynthesizer

HR = "confluence:hr"


def _seed(db: str) -> str:
    retriever = InMemoryRetriever()
    retriever.add(HR, "Leave policy: full-time staff get 21 days of annual leave.")
    store = SQLiteCognitionStore(db)
    cache = SemanticCache(retriever, StubSynthesizer(), store=store)
    unit_id = cache.get("what is our leave policy?").unit_id
    store.close()
    return unit_id


def test_ls_and_show(tmp_path, capsys) -> None:
    db = str(tmp_path / "c.db")
    unit_id = _seed(db)

    assert main(["--db", db, "ls"]) == 0
    assert unit_id in capsys.readouterr().out

    assert main(["--db", db, "show", unit_id]) == 0
    out = capsys.readouterr().out
    assert "understanding" in out
    assert "21 days" in out          # raw evidence is shown
    assert HR in out                 # provenance artifact id is shown


def test_invalidate_and_dirty(tmp_path, capsys) -> None:
    db = str(tmp_path / "c.db")
    unit_id = _seed(db)

    assert main(["--db", db, "invalidate", HR]) == 0
    assert unit_id in capsys.readouterr().out

    assert main(["--db", db, "invalidate", "nope:123"]) == 0
    assert "no cached units" in capsys.readouterr().out

    assert main(["--db", db, "dirty", unit_id]) == 0
    assert "marked dirty" in capsys.readouterr().out


def test_evict_and_purge(tmp_path, capsys) -> None:
    db = str(tmp_path / "c.db")
    _seed(db)

    assert main(["--db", db, "purge"]) == 1          # refuses without --yes
    assert "refusing" in capsys.readouterr().out

    assert main(["--db", db, "purge", "--yes"]) == 0
    assert "purged 1" in capsys.readouterr().out

    assert main(["--db", db, "ls"]) == 0
    assert "(cache is empty)" in capsys.readouterr().out


def test_stats(tmp_path, capsys) -> None:
    db = str(tmp_path / "c.db")
    _seed(db)
    assert main(["--db", db, "stats"]) == 0
    out = capsys.readouterr().out
    assert "units:" in out
    assert "tracked artifacts:" in out
