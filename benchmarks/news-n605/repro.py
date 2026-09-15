"""repro.py — re-derive the published news-benchmark numbers end to end.

One script, two arms, one grader (grader.py):

  POOL ARM   build the 609-article store with the documented recipe (one extractive unit
             per article), then serve the 605 held-out questions through the library's
             pool read path (``read_path="pool"``, serve_budget=1000) and grade answers.
             ``--header meta`` (default) reproduces the published headline point
             (0.731 @ ~981 ctx tokens, gold rank p50/p75/p90 = 1/6/15);
             ``--header default`` runs the shipping default header (measured ≈0.68 on the
             same store/questions — the delta is attribution metadata in the header).
  NAIVE ARM  dense top-k retrieval over the SAME chunks with the SAME metadata headers,
             k = 4/6/9/12, answered by the same model and graded by the same grader —
             the published naive curve (k4 0.585@~588 · k9 0.711@~1311 · k12 0.731@~1729).

Cost (honest): a full run — store build (~$1.5, one-time, resumable) + embeddings (~$0.03)
+ pool arm (~$0.3) + all four naive arms (~$1.1) — is ~$2-3 of OpenAI spend. Answers are
cached on disk, so re-runs and extra arms only pay for what is new. ``--limit 25`` smokes
the mechanics for a few cents (its numbers are meaningless). ``--grade-only`` re-grades
saved results with zero API calls.

Setup:
  pip install "coalent[fast,openai]"
  export OPENAI_API_KEY=sk-...
  # download corpus.json (see README.md) into ./data/
  python repro.py                     # build + pool(meta) + naive k=4,6,9,12
  python repro.py --arm pool --header both
  python repro.py --grade-only

Determinism note: the store build is LLM extraction — two independently built stores
differ by a few points (measured; see README). The grader and serving are deterministic
given a store; answers are near-deterministic. Judge re-runs against the confidence
intervals, not the third decimal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from coalent import LLMSynthesizer, OpenAIProvider, SemanticCache
from coalent.semantic import Chunk, OpenAIEmbedder
from coalent.semantic.serde import cognition_from_dict, cognition_to_dict
from grader import grade_rows, present, toks

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------- the documented recipe
CHUNK_SENTS = 3          # sentences per chunk
MAX_PASSAGES = 48        # chunk cap per article
SERVE_BUDGET = 1000      # pool serve budget (tokens, len//4 accounting)
BUILD_MODEL = "gpt-4o-mini"
BUILD_MAX_TOKENS = 2400
BUILD_DEPTH = 1.0
ANSWER_MODEL = "gpt-4.1-mini"
HIT_THRESHOLD = 0.33     # OpenAI-embedder calibration used by the published runs
COVERAGE_FLOOR = 0.28
RESIDUAL_FLOOR = 0.28    # retain source spans the extractor missed

# The extraction instruction of the published store — stance/attribution-preserving,
# query-independent. Passed verbatim to LLMSynthesizer(instruction=...).
EXTRACT_INSTRUCTION = (
    "IGNORE the question. EXTRACT every atomic fact the sources state, as a flat list of "
    "`claims`, one claim per fact. PRESERVE ATTRIBUTION AND STANCE: when a source, article, "
    "person, or organization asserts, argues, suggests, denies, criticizes, or predicts "
    "something, write the claim as '<WHO> <stance-verb> that <fact>' using the source's own "
    "stance verb - never flatten a quoted or attributed statement into a bare fact. Keep "
    "direct quotes as quotes with the speaker named. For EVERY number, include the number, "
    "exactly what it measures, its unit, and any condition attached. Include every named "
    "entity, quantity, date, limit, rate, and stated condition. Write each claim as one "
    "self-contained sentence. Do NOT summarize, do NOT omit any number, condition, or "
    "attribution. Copy values and quoted words verbatim."
    " When the source explicitly contrasts, concedes, or sequences two facts (however, but, "
    "while, despite, before/after), keep BOTH facts in ONE claim joined by the source's "
    "connective (e.g., 'Although X, Y'). Never split a contrast pair into separate lines."
)

# The answer prompt of every published accuracy figure (both arms, verbatim).
ANSWER_SYSTEM = (
    "Answer using ONLY the context. Reply with just the exact name, phrase, Yes, or No. "
    "If the context does not contain the answer, reply exactly 'none'."
)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [p.strip() for p in _SENT_SPLIT.split(text.strip()) if len(p.strip()) > 1]


def chunk_body(body: str) -> list[str]:
    sents = split_sentences(body)
    out = [" ".join(sents[i:i + CHUNK_SENTS]).strip() for i in range(0, len(sents), CHUNK_SENTS)]
    return [p for p in out if p][:MAX_PASSAGES]


def iter_jsonl(path: Path) -> Any:
    if not path.exists():
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception:
            continue  # torn line from an interrupted run: the record is re-earned


def _openai_client() -> Any:
    from openai import OpenAI
    return OpenAI()   # reads OPENAI_API_KEY


class MemoEmbedder(OpenAIEmbedder):
    """OpenAI-class embedder (so the cache resolves its OpenAI-calibrated defaults) with an
    in-memory memo: every text is embedded exactly once per run, batched where possible."""

    def __init__(self) -> None:
        # deliberately NOT calling super().__init__ — delegation, not a second client
        self._inner = OpenAIEmbedder(client=_openai_client())
        self._memo: dict[str, tuple[float, ...]] = {}

    def embed(self, text: str) -> list[float]:  # type: ignore[override]
        v = self._memo.get(text)
        if v is None:
            v = tuple(self._inner.embed(text))
            self._memo[text] = v
        return list(v)

    def embed_many(self, texts: list[str]) -> list[list[float]]:  # type: ignore[override]
        missing = [t for t in dict.fromkeys(texts) if t not in self._memo]
        for i in range(0, len(missing), 256):
            batch = missing[i:i + 256]
            for t, v in zip(batch, self._inner.embed_many(batch)):
                self._memo[t] = tuple(v)
            if len(missing) > 1000:
                print(f"    embedded {min(i + 256, len(missing))}/{len(missing)}", flush=True)
        return [list(self._memo[t]) for t in texts]


class DenseRetriever:
    """Dense top-k over the shared passage embeddings — the retriever of BOTH arms."""

    def __init__(self, passages: list[str], art_of: list[int], matrix: Any,
                 emb: MemoEmbedder, k: int = 12) -> None:
        self.passages, self.art_of, self.P, self.emb, self.k = passages, art_of, matrix, emb, k

    def top(self, query: str, k: int) -> list[int]:
        qe = np.asarray(self.emb.embed(query), dtype=np.float32)
        qe /= np.linalg.norm(qe) + 1e-9
        sims = self.P @ qe
        idx = np.argpartition(-sims, min(k, len(self.passages) - 1))[:k]
        return [int(i) for i in idx[np.argsort(-sims[idx])]]

    def retrieve(self, query: str, namespace: str | None = None) -> list[Chunk]:
        return [Chunk(artifact_id=f"art:{self.art_of[i]}", text=self.passages[i])
                for i in self.top(query, self.k)]


class _NoBuildSynth:
    """Serving is replayed over the warm store at $0 LLM spend: any synthesis attempt
    aborts that read loudly instead of silently paying for a build."""

    def synthesize(self, query: str, chunks: list[Chunk]) -> Any:
        raise RuntimeError("synthesis attempted during serve replay (warm store expected)")


class AnswerBank:
    """Disk-cached answer calls keyed by sha1(model|system|prompt) — resume-safe, and the
    backing store for --grade-only."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._bank: dict[str, str] = {}
        for d in iter_jsonl(path):
            self._bank[d["key"]] = d["text"]

    def ask(self, ctx: str, question: str) -> str:
        prompt = f"CONTEXT:\n{ctx}\n\nQUESTION: {question}"
        key = hashlib.sha1(f"{ANSWER_MODEL}|{ANSWER_SYSTEM}|{prompt}".encode()).hexdigest()
        if key in self._bank:
            return self._bank[key]
        delay = 2.0
        for attempt in range(8):
            try:
                r = _openai_client().chat.completions.create(
                    model=ANSWER_MODEL, max_completion_tokens=2048,
                    messages=[{"role": "system", "content": ANSWER_SYSTEM},
                              {"role": "user", "content": prompt}])
                text = (r.choices[0].message.content or "").strip()
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "text": text}) + "\n")
                self._bank[key] = text
                return text
            except Exception:
                if attempt == 7:
                    return ""
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
        return ""


# ----------------------------------------------------------------------- world loading
def load_corpus(data_dir: Path) -> list[dict[str, Any]]:
    path = data_dir / "corpus.json"
    if not path.exists():
        sys.exit(f"{path} not found — download the MultiHopRAG corpus first (see README.md)")
    corpus = json.load(open(path, encoding="utf-8"))
    if len(corpus) != 609:
        sys.exit(f"corpus.json has {len(corpus)} articles, expected 609 — wrong file?")
    return corpus


def load_questions() -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in open(HERE / "questions.jsonl", encoding="utf-8")]
    if len(rows) != 605:
        sys.exit(f"questions.jsonl has {len(rows)} rows, expected 605")
    return rows


def build_passages(corpus: list[dict[str, Any]]) -> tuple[list[str], list[int]]:
    passages: list[str] = []
    art_of: list[int] = []
    for ai, art in enumerate(corpus):
        for p in chunk_body(art.get("body", "")):
            passages.append(p)
            art_of.append(ai)
    return passages, art_of


def passage_matrix(passages: list[str], emb: MemoEmbedder, out_dir: Path) -> Any:
    """Embed all passages once; persist the matrix so later runs and arms are free."""
    cache_path = out_dir / "passages-emb.npy"
    if cache_path.exists():
        P = np.load(cache_path)
        if P.shape[0] == len(passages):
            for p, v in zip(passages, P):
                emb._memo[p] = tuple(float(x) for x in v)
            return P
        print(f"  {cache_path.name} shape {P.shape} does not match {len(passages)} passages "
              f"— re-embedding", flush=True)
    vecs = emb.embed_many(passages)
    P = np.asarray(vecs, dtype=np.float32)
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-9
    np.save(cache_path, P)
    return P


def metadata_headers(corpus: list[dict[str, Any]]) -> list[str]:
    return [f"[{a.get('title', '?')} | {a.get('source', '?')} | "
            f"{str(a.get('published_at', '?'))[:10]}]" for a in corpus]


# ----------------------------------------------------------------------- store build
def build_store(corpus: list[dict[str, Any]], emb: MemoEmbedder, out_dir: Path,
                workers: int = 6) -> Path:
    """One extractive unit per article, appended to disk on completion — a crash or re-run
    never repeats paid work. This is the exact published recipe."""
    path = out_dir / "units.jsonl"
    done: dict[int, dict[str, Any]] = {}
    for d in iter_jsonl(path):
        if d.get("understanding", {}).get("_synthesis_failed"):
            continue                          # failed synthesis is retried on resume
        done[d["_article"]] = d
    todo = [i for i in range(len(corpus)) if i not in done]
    print(f"store: {len(done)} units cached, {len(todo)} to build -> {path}", flush=True)
    if not todo:
        return path

    def build_one(ai: int) -> dict[str, Any]:
        chunks = [Chunk(artifact_id=f"art:{ai}", text=p)
                  for p in chunk_body(corpus[ai].get("body", ""))]

        class _One:
            def retrieve(self, query: str, namespace: str | None = None) -> list[Chunk]:
                return chunks

        synth = LLMSynthesizer(OpenAIProvider(client=_openai_client()), model=BUILD_MODEL,
                               max_tokens=BUILD_MAX_TOKENS, depth=BUILD_DEPTH,
                               instruction=EXTRACT_INSTRUCTION)
        # read_path="unit" pinned (v0.7 flips the resolved default to pool under a
        # semantic embedder): this throwaway cache exists only to BUILD one unit with
        # the frozen store recipe — the published store was produced on this exact path.
        cache = SemanticCache(_One(), synth, embedder=emb, hit_threshold=HIT_THRESHOLD,
                              coverage_floor=COVERAGE_FLOOR, residual_floor=RESIDUAL_FLOOR,
                              read_path="unit")
        delay = 5.0
        for attempt in range(6):
            try:
                r = cache.get(f"key facts of the article: {corpus[ai]['title']}")
                u = cache._units[r.unit_id].understanding
                if not (u.get("claims") or u.get("summary") or u.get("facts")):
                    raise ValueError("hollow unit (no claims/summary/facts)")
                break
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
        d = cognition_to_dict(cache._units[r.unit_id])
        d["_article"] = ai
        return d

    failures: list[int] = []
    t0 = time.perf_counter()
    with open(path, "a", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(build_one, ai): ai for ai in todo}
        n = 0
        for fut in as_completed(futs):
            try:
                d = fut.result()
            except Exception as exc:
                failures.append(futs[fut])
                print(f"  build FAILED art:{futs[fut]} ({type(exc).__name__}) — "
                      f"re-run resumes it", flush=True)
                continue
            f.write(json.dumps(d) + "\n")
            f.flush()
            n += 1
            if n % 25 == 0:
                rate = (time.perf_counter() - t0) / n
                print(f"  built {n}/{len(todo)} (~{rate:.1f}s/unit, "
                      f"eta {rate * (len(todo) - n) / 60:.0f} min)", flush=True)
    if failures:
        sys.exit(f"{len(failures)} articles failed to build ({failures[:10]}...) — "
                 f"re-run to retry them before serving")
    return path


def load_units(path: Path) -> list[Any]:
    units = []
    for d in iter_jsonl(path):
        if d.get("understanding", {}).get("_synthesis_failed"):
            continue
        units.append(cognition_from_dict({k: v for k, v in d.items() if not k.startswith("_")}))
    return units


# ----------------------------------------------------------------------- the two arms
def run_pool(questions: list[dict[str, Any]], units: list[Any], retriever: DenseRetriever,
             emb: MemoEmbedder, corpus: list[dict[str, Any]], header_mode: str,
             bank: AnswerBank, out_dir: Path, workers: int) -> Path:
    """Serve all questions through the library pool read path over the warm store."""
    heads = metadata_headers(corpus)

    def meta_header(u: Any) -> str:
        for c in u.evidence:
            if c.artifact_id.startswith("art:"):
                ai = int(c.artifact_id.split(":")[1])
                if ai < len(heads):
                    return heads[ai]
        return ""

    header = meta_header if header_mode == "meta" else None
    cache = SemanticCache(retriever, _NoBuildSynth(), embedder=emb, read_path="pool",
                          serve_budget=SERVE_BUDGET, pool_header=header)
    # Bulk-load the store through the same maps the serde round-trip uses.
    for u in units:
        cache._units[u.id] = u
        cache._reindex(u)

    out_path = out_dir / f"results-pool-{header_mode}.jsonl"
    have = {d["query"] for d in iter_jsonl(out_path)}
    todo = [q for q in questions if q["query"] not in have]
    print(f"pool[{header_mode}]: {len(have)} served, {len(todo)} to serve", flush=True)
    if not todo:
        return out_path

    emb.embed_many([q["query"] for q in todo])       # batch the query embeddings
    rows: list[dict[str, Any]] = []
    aborted = 0
    for q in todo:
        try:
            r = cache.get(q["query"])
        except RuntimeError:
            aborted += 1                              # gate miss tried to build: counted, $0
            rows.append({"arm": f"pool-{header_mode}", "query": q["query"],
                         "qtype": q["qtype"], "gold": q["gold"], "ctx": "",
                         "ctx_tok": 0, "rank": None, "served": False})
            continue
        payload = (r.context or {}).get("pool", "") if isinstance(r.context, dict) else ""
        rank = None
        for j, rc in enumerate(r.pool or []):
            if present(q["gold"], rc.claim):
                rank = j + 1
                break
        rows.append({"arm": f"pool-{header_mode}", "query": q["query"], "qtype": q["qtype"],
                     "gold": q["gold"], "ctx": payload, "ctx_tok": toks(payload),
                     "rank": rank, "served": True,
                     "gold_in_ctx": present(q["gold"], payload)})
    if aborted:
        print(f"  {aborted} reads aborted on a build attempt (expected 0 on a full store)",
              flush=True)

    _answer_rows(rows, bank, workers)
    with open(out_path, "a", encoding="utf-8") as f:
        for row in rows:
            row = dict(row)
            row.pop("ctx", None)                      # keep result rows small; ctx_tok stays
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out_path


def run_naive(questions: list[dict[str, Any]], retriever: DenseRetriever,
              emb: MemoEmbedder, corpus: list[dict[str, Any]], k_list: list[int],
              bank: AnswerBank, out_dir: Path, workers: int) -> list[Path]:
    """The baseline arm: dense top-k over the same chunks, same headers, same grader."""
    heads = metadata_headers(corpus)
    emb.embed_many([q["query"] for q in questions])
    paths = []
    for k in k_list:
        out_path = out_dir / f"results-naive-k{k}.jsonl"
        have = {d["query"] for d in iter_jsonl(out_path)}
        todo = [q for q in questions if q["query"] not in have]
        print(f"naive k={k}: {len(have)} done, {len(todo)} to run", flush=True)
        if not todo:
            paths.append(out_path)
            continue
        rows = []
        for q in todo:
            idx = retriever.top(q["query"], k)
            ctx = "\n\n".join(f"{heads[retriever.art_of[i]]}\n{retriever.passages[i]}"
                              for i in idx)
            rows.append({"arm": f"naive-k{k}", "query": q["query"], "qtype": q["qtype"],
                         "gold": q["gold"], "ctx": ctx, "ctx_tok": toks(ctx),
                         "gold_in_ctx": present(q["gold"], ctx)})
        _answer_rows(rows, bank, workers)
        with open(out_path, "a", encoding="utf-8") as f:
            for row in rows:
                row = dict(row)
                row.pop("ctx", None)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        paths.append(out_path)
    return paths


def _answer_rows(rows: list[dict[str, Any]], bank: AnswerBank, workers: int) -> None:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(bank.ask, r.get("ctx", ""), r["query"]): r for r in rows}
        done = 0
        for fut in as_completed(futs):
            r = futs[fut]
            r["answer"] = fut.result()[:300]
            done += 1
            if done % 100 == 0:
                print(f"  answered {done}/{len(rows)}", flush=True)


# ----------------------------------------------------------------------- reporting
def summarize(path: Path) -> None:
    rows = list(iter_jsonl(path))
    if not rows:
        print(f"  {path.name}: no rows")
        return
    m = grade_rows(rows)
    lo, hi = m["strict_ci95"]
    line = (f"  {path.stem.removeprefix('results-'):<14} n={m['n']:<4} "
            f"strict {m['strict']}/{m['n']} = {m['strict_rate']:.4f} "
            f"CI95 [{lo:.4f},{hi:.4f}]  exact {m['exact_rate']:.4f}  "
            f"refusals {m['refusal_rate']:.4f}")
    if "ctx_tok_mean" in m:
        line += f"  ctx-tok mean {m['ctx_tok_mean']:.0f} / median {m['ctx_tok_median']}"
    print(line)
    ranks = sorted(r["rank"] for r in rows if r.get("rank"))
    if ranks:
        def pctl(p: float) -> int:
            return ranks[min(len(ranks) - 1, max(0, int(p * len(ranks) + 0.999) - 1))]
        print(f"  {'':<14} gold rank p50/p75/p90 = {pctl(0.50)}/{pctl(0.75)}/{pctl(0.90)}  "
              f"(not in served claims: {sum(1 for r in rows if not r.get('rank'))})")
    if any("gold_in_ctx" in r for r in rows):
        g = sum(bool(r.get("gold_in_ctx")) for r in rows)
        print(f"  {'':<14} gold present in served context: {g}/{m['n']} = {g / m['n']:.4f}")


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=HERE / "data",
                    help="directory containing corpus.json (default: ./data)")
    ap.add_argument("--out", type=Path, default=HERE / "out",
                    help="output directory (store, caches, results)")
    ap.add_argument("--arm", choices=("pool", "naive", "all"), default="all")
    ap.add_argument("--header", choices=("meta", "default", "both"), default="meta",
                    help="pool header arm: 'meta' = [title | source | date] (the published "
                         "headline), 'default' = the library's shipping default header")
    ap.add_argument("--k", default="4,6,9,12", help="naive top-k list (comma-separated)")
    ap.add_argument("--limit", type=int, default=0, help="run only the first N questions "
                    "(cheap smoke; numbers are meaningless)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--grade-only", action="store_true",
                    help="re-grade saved results; zero API calls")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.grade_only:
        # grade_rows re-applies the grader to the SAVED answers — zero API calls.
        print("== saved results (re-graded from saved answers) ==")
        for path in sorted(args.out.glob("results-*.jsonl")):
            summarize(path)
        return

    corpus = load_corpus(args.data)
    questions = load_questions()
    if args.limit:
        questions = questions[:args.limit]
        print(f"SMOKE: first {len(questions)} questions only — numbers are meaningless")

    emb = MemoEmbedder()
    passages, art_of = build_passages(corpus)
    print(f"world: {len(corpus)} articles, {len(passages)} passages")
    P = passage_matrix(passages, emb, args.out)
    retriever = DenseRetriever(passages, art_of, P, emb, k=12)
    bank = AnswerBank(args.out / "answers.jsonl")

    result_paths: list[Path] = []
    if args.arm in ("pool", "all"):
        units_path = build_store(corpus, emb, args.out)
        units = load_units(units_path)
        print(f"store: {len(units)} units, "
              f"{sum(len(u.understanding.get('claims') or []) for u in units)} claims")
        modes = ("meta", "default") if args.header == "both" else (args.header,)
        for mode in modes:
            result_paths.append(run_pool(questions, units, retriever, emb, corpus, mode,
                                         bank, args.out, args.workers))
    if args.arm in ("naive", "all"):
        k_list = [int(x) for x in args.k.split(",") if x.strip()]
        result_paths.extend(run_naive(questions, retriever, emb, corpus, k_list,
                                      bank, args.out, args.workers))

    print("\n== results (strict grader; see README.md for the published anchors) ==")
    for path in result_paths:
        summarize(path)


if __name__ == "__main__":
    main()
