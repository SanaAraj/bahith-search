"""Measure retrieval latency and (optionally) answer-generation latency + cost.

Retrieval latency is always measured. The answer-generation path is measured
only when ``OPENAI_API_KEY`` is set, since it makes live LLM calls; per-query
cost is estimated from the reported token usage and the configured per-token
rates. Results are written to ``bench/results/<timestamp>.json`` with the model
identifiers and run date so the README table stays reproducible and dated.

Usage::

    python benchmark.py                 # retrieval only (no key) or full (key set)
    python benchmark.py --iterations 5  # repeats per query for retrieval
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import generate
import search
from config import (
    EMBEDDING_MODEL,
    LLM_INPUT_COST_PER_1M,
    LLM_OUTPUT_COST_PER_1M,
    MODEL_NAME,
    OPENAI_API_KEY,
)

BENCH_DIR = Path(__file__).resolve().parent / "bench"
RESULTS_DIR = BENCH_DIR / "results"
DATASET_PATH = Path(__file__).resolve().parent / "eval" / "dataset.json"


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (pct in [0, 100])."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _summary(latencies_ms: list[float]) -> dict:
    # 3 decimals so sub-millisecond paths (BM25) don't collapse to 0.0.
    return {
        "p50_ms": round(_percentile(latencies_ms, 50), 3),
        "p95_ms": round(_percentile(latencies_ms, 95), 3),
        "mean_ms": round(statistics.fmean(latencies_ms), 3),
        "n": len(latencies_ms),
    }


def load_queries() -> list[str]:
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    return [row["query"] for row in data]


def benchmark_retrieval(queries: list[str], iterations: int) -> dict:
    results = {}
    for mode in ("semantic", "keyword", "hybrid"):
        # Warm up (model load, caches) so it doesn't skew the first sample.
        search.search(queries[0], mode=mode, top_k=5)
        latencies = []
        for _ in range(iterations):
            for q in queries:
                start = time.perf_counter()
                search.search(q, mode=mode, top_k=5)
                latencies.append((time.perf_counter() - start) * 1000)
        results[mode] = _summary(latencies)
    return results


def benchmark_generation(queries: list[str]) -> dict | None:
    if not OPENAI_API_KEY:
        return None

    latencies, costs, in_tokens, out_tokens = [], [], [], []
    client = generate.get_client()

    for q in queries:
        hits = search.search(q, mode="hybrid", top_k=5)
        context_parts = [f"[{i}] {r['title']}:\n{r['content'][:500]}" for i, r in enumerate(hits[:3], 1)]
        messages = [
            {"role": "system", "content": generate.SYSTEM_PROMPT},
            {"role": "user", "content": "السياق:\n" + "\n\n".join(context_parts) + f"\n\nالسؤال: {q}\n\nالإجابة:"},
        ]
        start = time.perf_counter()
        resp = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, temperature=0.3, max_tokens=400
        )
        latencies.append((time.perf_counter() - start) * 1000)

        usage = resp.usage
        pt, ct = usage.prompt_tokens, usage.completion_tokens
        in_tokens.append(pt)
        out_tokens.append(ct)
        costs.append(pt / 1e6 * LLM_INPUT_COST_PER_1M + ct / 1e6 * LLM_OUTPUT_COST_PER_1M)

    summary = _summary(latencies)
    summary.update(
        {
            "avg_input_tokens": round(statistics.fmean(in_tokens), 1),
            "avg_output_tokens": round(statistics.fmean(out_tokens), 1),
            "cost_per_query_usd": round(statistics.fmean(costs), 6),
            "input_cost_per_1m_usd": LLM_INPUT_COST_PER_1M,
            "output_cost_per_1m_usd": LLM_OUTPUT_COST_PER_1M,
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Bahith latency/cost benchmark")
    parser.add_argument("--iterations", type=int, default=5, help="Retrieval repeats per query")
    parser.add_argument("--save", action="store_true", help="Write JSON report to bench/results/")
    args = parser.parse_args()

    queries = load_queries()
    report = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "hardware": f"{platform.system()} {platform.machine()} (CPU)",
        "python": platform.python_version(),
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": MODEL_NAME if OPENAI_API_KEY else None,
        "num_queries": len(queries),
        "retrieval_iterations": args.iterations,
        "retrieval": benchmark_retrieval(queries, args.iterations),
        "generation": benchmark_generation(queries),
    }

    print(f"Bahith benchmark  ({report['timestamp']})")
    print(f"  hardware        : {report['hardware']}")
    print(f"  embedding model : {report['embedding_model']}")
    print(f"  queries         : {report['num_queries']}  x{args.iterations} iterations")
    print("  retrieval latency (ms):")
    for mode, s in report["retrieval"].items():
        print(f"    {mode:9s} p50={s['p50_ms']:.3f}  p95={s['p95_ms']:.3f}  mean={s['mean_ms']:.3f}")
    gen = report["generation"]
    if gen:
        print("  answer generation:")
        print(f"    latency  p50={gen['p50_ms']:.1f}ms  p95={gen['p95_ms']:.1f}ms")
        print(f"    cost/query=${gen['cost_per_query_usd']:.6f}  "
              f"(in={gen['avg_input_tokens']:.0f} out={gen['avg_output_tokens']:.0f} tokens)")
    else:
        print("  answer generation: skipped (set OPENAI_API_KEY to measure latency + cost)")

    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / f"bench_{report['timestamp'].replace(':', '')}.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  saved report    : {out}")


if __name__ == "__main__":
    main()
