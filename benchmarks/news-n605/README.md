# news-n605 — the reproduction package for Coalent's news-benchmark numbers

Everything needed for a stranger to re-derive the published headline numbers from scratch:
the frozen 605 questions with gold answers, the exact grader, the corpus recipe, and one
script that builds the store, runs both arms, and grades them end to end.

What this re-derives (n=605, strict grader, answer model `gpt-4.1-mini`):

| arm | accuracy | mean ctx tokens | notes |
|---|---|---|---|
| pool, metadata header (`--header meta`) | **0.731** | ~981 | the published headline point; gold rank p50/p75/p90 = 1/6/15 |
| pool, shipping default header (`--header default`) | ~0.68 | ~985 | same store, library default header — see "the two header arms" |
| naive top-k, k=4 | 0.585 | ~588 | |
| naive top-k, k=9 | 0.711 | ~1311 | an independent re-run measured 0.717 @ ~1306 (harness stability) |
| naive top-k, k=12 | 0.731 | ~1729 | naive's best measured configuration |

(A k=3 point was also measured at 0.557 @ ~444. k=6 is runnable here but was not a
published anchor.) The headline claim these numbers ground: **the pool read path matches
naive dense RAG's best measured accuracy (0.731) at ~43% fewer context tokens per read**
(981 vs 1729), with per-claim source attribution and freshness instead of raw chunks.

## The dataset and its license

Questions and articles come from **MultiHop-RAG** (Tang & Yang, 2024 — "MultiHop-RAG:
Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries"), a third-party
benchmark of multi-hop questions over 609 real news articles:

- GitHub: <https://github.com/yixuantt/MultiHop-RAG>
- Hugging Face: <https://huggingface.co/datasets/yixuantt/MultiHopRAG>

The dataset is released under **ODC-BY 1.0** (Open Data Commons Attribution), which
permits copying, redistribution, and adaptation of the dataset **with attribution** —
which is why `questions.jsonl` (the 605-question subset with gold answers, attributed
here to the MultiHop-RAG authors) ships in this repo. The **article bodies are not
re-hosted here**: the articles are news content whose underlying copyright belongs to the
original outlets, and a dataset license cannot transfer that, so we point you at the
official distribution instead of redistributing it ourselves.

**Corpus prep:** download `corpus.json` (609 articles, ~5 MB) from the official
MultiHop-RAG repository (`dataset/corpus.json`) or the Hugging Face dataset, and place it
at `./data/corpus.json`. That single file is the whole corpus — there is no article
selection step; all 609 are used, and article index `art:N` = position in `corpus.json`.

## Files

- `questions.jsonl` — the frozen 605 held-out questions: `{"query", "gold", "qtype"}`
  per line (223 inference / 204 comparison / 178 temporal; no unanswerable nulls). This
  exact set, in this exact order, is the population behind every number above.
- `grader.py` — the strict grader, standalone (stdlib only). Every published accuracy
  figure is `grader.present(gold, answer)`: normalized containment with a word-boundary
  guard for short golds. Import it or run `python grader.py results.jsonl`.
- `repro.py` — build + serve + answer + grade, both arms, resumable, `--grade-only`.

## Running it

```bash
pip install "coalent[fast,openai]"
export OPENAI_API_KEY=sk-...
# put corpus.json at ./data/corpus.json (see above)

python repro.py                          # build store + pool(meta) + naive k=4,6,9,12
python repro.py --arm pool --header both # both pool header arms
python repro.py --limit 25               # mechanics smoke, a few cents (numbers meaningless)
python repro.py --grade-only             # re-grade saved results, zero API calls
```

**Cost, honestly:** a full run is **~$2–3** of OpenAI spend — store build ~$1.5 (one-time,
609 `gpt-4o-mini` extraction calls; resumable, crash-safe), embeddings ~$0.03
(`text-embedding-3-small`, cached to disk), pool arm ~$0.30, all four naive arms ~$1.10
(`gpt-4.1-mini` answers, cached to disk so re-runs are free). Wall clock ~30–60 min,
dominated by the build.

## The recipe (what `repro.py` encodes)

- **Chunking:** sentence split on `.!?`, 3 sentences per chunk, max 48 chunks/article.
  Both arms retrieve over the SAME chunks with the same embeddings.
- **Store build (pool arm):** one extractive unit per article — `LLMSynthesizer`
  (`gpt-4o-mini`, `max_tokens=2400`, `depth=1.0`) with the stance/attribution-preserving
  extraction instruction (verbatim in `repro.py`), build query
  `"key facts of the article: {title}"`, `hit_threshold=0.33`, `coverage_floor=0.28`,
  `residual_floor=0.28`.
- **Serving (pool arm):** the library pool read path — `read_path="pool"`,
  `serve_budget=1000`, shipping-default gates — over the warm store. Serving is replayed
  at $0 LLM spend (a synthesis attempt aborts that read loudly rather than paying).
- **Naive arm:** dense top-k over the same chunks, `k ∈ {4,6,9,12}`, each chunk prefixed
  with the same `[title | source | date]` metadata header (both arms get metadata — the
  comparison is fair by construction).
- **Answering + grading (identical across arms):** `gpt-4.1-mini`, the exact answer
  prompt in `repro.py` (context-only, refuse with 'none'), graded by `grader.present`.

## The two header arms — which is which

`pool_header` controls the per-source attribution line above each unit's claims in the
served context:

- `--header meta` — the harness supplies `[title | source | date]` from corpus metadata.
  **This is the arm behind the published 0.731.** The store predates metadata capture at
  ingest, so the header is supplied by the harness callable — exactly as the original run
  did.
- `--header default` — `pool_header=None`, i.e. the library's shipping default (a `## `
  header derived from the unit's build query). Measured ≈0.68 on the same store and
  questions. The ~5-point gap is attribution metadata in the header, not retrieval or
  extraction; carrying source metadata into the default header is tracked work.

## Caveats that keep the numbers honest

- **Amortization:** the ~43%-fewer-tokens figure compares per-read serving cost against
  naive k=12 on a warm store. Building the store costs tokens up front: on a **cold**
  single-pass run (build everything, read each question once) total spend is only ~18%
  below naive. The 43% figure is reached once each source is read **≥4–5 times** — the
  reuse regime the cache is for. Below that read rate, naive RAG is the cheaper tool.
- **Store variance:** extraction is an LLM call — two independently built stores differ
  by a few accuracy points (measured). The naive arm re-ran within ±0.006. Compare your
  re-run against the 95% CIs `repro.py` prints (±~0.036 at n=605), not the third decimal.
- **Accuracy parity, not accuracy dominance:** at matched token budgets the pool path
  statistically ties naive dense RAG on this benchmark — the claim is equal accuracy at
  fewer serving tokens plus attribution/freshness, never "beats RAG on accuracy".
- **Grader:** strict containment (`grader.py`) — not an LLM judge. Its word-boundary
  guard makes short golds ("no", "yes") hard to match by luck; long golds must appear
  verbatim after normalization.

## Attribution

MultiHop-RAG: Yixuan Tang and Yi Yang. "MultiHop-RAG: Benchmarking Retrieval-Augmented
Generation for Multi-Hop Queries" (2024). Dataset © its authors, ODC-BY 1.0. The 605
questions in `questions.jsonl` are an attributed subset; article text is not included.
