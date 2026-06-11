"""An honest mechanism benchmark for the cognitive cache.

Measures the cheap + fresh triangle on a multi-document corpus under churn, with
an INDEPENDENT correctness oracle (the harness's own ground truth — not the
cache's hashes) and real token cost. Compares Coalent against:

  * **NaiveRAG**   — re-retrieves on every read (always fresh, full cost).
  * **StaleCache** — a semantic cache WITHOUT provenance invalidation (cheap, but
    goes stale on source change). The "what we'd be without the moat" baseline.

Expected, honest result: Coalent matches NaiveRAG's freshness (zero stale) at far
below its cost, while StaleCache is cheap but stale.

This is a deterministic *mechanism* benchmark (no LLM) — it proves the
freshness/cost behaviour. A real-LLM answer-quality benchmark vs GraphRAG on a
real dataset is a separate, API-key-gated artifact.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from ..semantic import Chunk, HashingEmbedder, SemanticCache, StubSynthesizer
from ..semantic.embedding import cosine

TOPICS = ["leave", "remote", "security", "payroll", "expenses"]
CHANGED = {"leave": 1, "security": 1}  # these two sources change mid-run


def token_count(text: str) -> int:
    return len(text.split())


def _doc(topic: str, version: int) -> str:
    return f"the {topic} policy allowance is currently mk{topic}{version}"


def _query(topic: str) -> str:
    # Distinct per topic so each becomes its own cognition unit (the queries must
    # not collapse into one under the embedder used).
    return f"{topic} policy"


def _marker(topic: str, version: int) -> str:
    return f"mk{topic}{version}"


class CorpusRetriever:
    """Top-k retrieval over a LIVE (mutable) corpus dict the harness owns."""

    def __init__(self, corpus: dict[str, str], *, top_k: int = 1) -> None:
        self._corpus = corpus
        self._embedder = HashingEmbedder()
        self._top_k = top_k

    def retrieve(self, query: str, *, namespace: str | None = None) -> list[Chunk]:
        qe = self._embedder.embed(query)
        scored: list[tuple[float, str, str]] = []
        for artifact_id, text in self._corpus.items():
            score = cosine(qe, self._embedder.embed(text))
            if score > 0.0:
                scored.append((score, artifact_id, text))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [Chunk(artifact_id, text) for _, artifact_id, text in scored[: self._top_k]]


@dataclass(slots=True)
class Report:
    system: str
    reads: int = 0
    correct: int = 0
    stale: int = 0
    cost_tokens: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.reads if self.reads else 0.0

    @property
    def stale_rate(self) -> float:
        return self.stale / self.reads if self.reads else 0.0


class _System(Protocol):
    def serve(self, query: str) -> tuple[str, int]:
        ...

    def on_change(self, artifact_id: str, text: str) -> None:
        ...


class _Naive:
    """Re-retrieves the live corpus on every read — always fresh, always pays."""

    def __init__(self, corpus: dict[str, str]) -> None:
        self._retriever = CorpusRetriever(corpus)

    def serve(self, query: str) -> tuple[str, int]:
        chunks = self._retriever.retrieve(query)
        context = " ".join(chunk.text for chunk in chunks)
        return context, sum(token_count(chunk.text) for chunk in chunks)

    def on_change(self, artifact_id: str, text: str) -> None:
        return None


class _Semantic:
    """A semantic cache. With ``invalidates=False`` it never hears about source
    changes (the StaleCache baseline); with ``True`` it is Coalent."""

    def __init__(self, corpus: dict[str, str], *, invalidates: bool) -> None:
        # coverage_floor=0 isolates the variable under test to invalidation alone.
        self._cache = SemanticCache(CorpusRetriever(corpus), StubSynthesizer(), coverage_floor=0.0)
        self._invalidates = invalidates

    def serve(self, query: str) -> tuple[str, int]:
        result = self._cache.get(query)
        context = f"{result.understanding.get('summary', '')} {result.raw_text}"
        cost = 0 if result.cache_hit else sum(token_count(c.text) for c in result.evidence)
        return context, cost

    def on_change(self, artifact_id: str, text: str) -> None:
        if self._invalidates:
            self._cache.source_changed(artifact_id, text=text)


def _score(system: _System, topic: str, current: dict[str, str], report: Report) -> None:
    context, cost = system.serve(_query(topic))
    report.reads += 1
    report.cost_tokens += cost
    current_marker = current[topic]
    if current_marker in context:
        report.correct += 1
    elif any(_marker(topic, v) in context for v in range(5) if _marker(topic, v) != current_marker):
        report.stale += 1  # an OLD value is served while the current one is absent


def run_benchmark() -> dict[str, Report]:
    """Run the cheap+fresh triangle and return a report per system."""
    builders: dict[str, Callable[[dict[str, str]], _System]] = {
        "NaiveRAG": lambda corpus: _Naive(corpus),
        "StaleCache": lambda corpus: _Semantic(corpus, invalidates=False),
        "Coalent": lambda corpus: _Semantic(corpus, invalidates=True),
    }
    reports: dict[str, Report] = {}
    for name, build in builders.items():
        corpus = {f"policy:{t}": _doc(t, 0) for t in TOPICS}
        current = {t: _marker(t, 0) for t in TOPICS}
        system = build(corpus)
        report = Report(system=name)

        for topic in TOPICS:  # phase 1 — cold reads (everyone correct, everyone pays)
            _score(system, topic, current, report)

        for topic, version in CHANGED.items():  # churn
            artifact_id = f"policy:{topic}"
            corpus[artifact_id] = _doc(topic, version)
            current[topic] = _marker(topic, version)
            system.on_change(artifact_id, corpus[artifact_id])

        for topic in TOPICS:  # phase 2 — warm reads (where the systems diverge)
            _score(system, topic, current, report)

        reports[name] = report
    return reports
