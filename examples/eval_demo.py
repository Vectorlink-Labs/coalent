"""Print the cheap + fresh comparison table — Coalent vs naive RAG vs a stale cache.

    python examples/eval_demo.py

Same query + source-change workload, scored by an independent oracle.
"""
from __future__ import annotations

from coalent.evaluation.harness import run_benchmark


def main() -> None:
    reports = run_benchmark()
    print(f"{'system':<12}{'accuracy':>10}{'stale_rate':>12}{'cost_tokens':>13}")
    print("-" * 47)
    for name, report in reports.items():
        print(f"{name:<12}{report.accuracy:>9.0%}{report.stale_rate:>12.0%}{report.cost_tokens:>13}")


if __name__ == "__main__":
    main()
