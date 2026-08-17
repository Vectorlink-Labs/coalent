"""The strict grader behind the published news-benchmark numbers — standalone, no deps.

This is the exact measurement instrument: every published accuracy figure (the pool-path
point and the naive top-k curve) is `present(gold, answer)` over the model's short answer.
Porting note: byte-for-byte the same normalization + containment rule the original runs
used, so a re-run here is graded on the same scale as the published numbers.

Definitions
-----------
- ``norm``     lowercase, strip punctuation, drop articles (the/a/an), collapse spaces.
- ``present``  normalized containment of the gold answer in the text, with a WORD-BOUNDARY
  guard for short golds ("no" must not match inside "november"). This is the primary
  ("strict") grade: it accepts "Sam Bankman-Fried." for gold "Sam Bankman-Fried" but
  rejects partial-token matches.
- ``exact``    normalized string equality — stricter, reported as an informational column.
- ``refusal``  the answer normalizes to "none"/"" (the answer prompt mandates 'none' when
  the context lacks the answer).
- ``toks``     len(text)//4 — the deliberately simple token accounting used for every
  published context-token figure (identical across arms, so ratios are exact).

Usage as a module::

    from grader import present, norm, toks, wilson

Usage as a CLI — grade a saved results file (one JSON object per line with at least
``gold`` and ``answer`` fields; ``ctx_tok`` is summarized when present)::

    python grader.py results.jsonl
"""
from __future__ import annotations

import json
import math
import re
import string
import sys

_PUNCT = str.maketrans("", "", string.punctuation)
_ARTICLES = re.compile(r"\b(the|a|an)\b")


def norm(s: object) -> str:
    """Lowercase, strip punctuation, drop English articles, collapse double spaces."""
    return _ARTICLES.sub(" ", str(s).lower().translate(_PUNCT)).replace("  ", " ").strip()


def present(gold: str, text: str) -> bool:
    """Is the gold answer present in the text — WORD-boundary for short answers, so
    'no' can't match inside 'november' and inflate the metric. THE strict grade."""
    g, t = norm(gold), norm(text)
    if not g:
        return False
    if len(g) <= 3 or g in ("yes", "no"):
        return g in t.split()
    return g in t


def exact(gold: str, text: str) -> bool:
    """Normalized string equality (informational; stricter than ``present``)."""
    return norm(text) == norm(gold)


def refusal(text: str) -> bool:
    """The answer model declined ('none' is mandated when the context lacks the answer)."""
    return norm(text) in ("none", "", "insufficient information")


def toks(text: str) -> int:
    """len//4 — the published token accounting (identical across arms, ratios exact)."""
    return max(1, len(text) // 4)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% score interval for k successes in n trials."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def grade_rows(rows: list[dict]) -> dict:
    """Grade result rows ({gold, answer[, ctx_tok]}) -> the summary metric dict."""
    n = len(rows)
    ks = sum(present(r["gold"], r.get("answer", "")) for r in rows)
    ke = sum(exact(r["gold"], r.get("answer", "")) for r in rows)
    kr = sum(refusal(r.get("answer", "")) for r in rows)
    lo, hi = wilson(ks, n)
    tok_vals = sorted(int(r["ctx_tok"]) for r in rows if "ctx_tok" in r)
    out = {
        "n": n,
        "strict": ks, "strict_rate": ks / n if n else 0.0, "strict_ci95": (lo, hi),
        "exact": ke, "exact_rate": ke / n if n else 0.0,
        "refusals": kr, "refusal_rate": kr / n if n else 0.0,
    }
    if tok_vals:
        out["ctx_tok_mean"] = sum(tok_vals) / len(tok_vals)
        out["ctx_tok_median"] = tok_vals[len(tok_vals) // 2]
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    rows = []
    for line in open(argv[0], encoding="utf-8"):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    m = grade_rows(rows)
    print(json.dumps(m, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
