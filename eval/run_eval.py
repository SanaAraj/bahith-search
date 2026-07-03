"""Run the retrieval mini-evaluation and report recall@5 and MRR.

The dataset (``eval/dataset.json``) is a fixed set of Arabic queries, each with
the document ``source`` files judged relevant. Relevance is defined at the
document level: a document counts as retrieved if any of its chunks appears in
the ranked results.

Usage::

    python -m eval.run_eval                       # print report
    python -m eval.run_eval --k 5 --min-recall 0.7 --min-mrr 0.6

Exits non-zero when a metric falls below its threshold, so CI can gate on
retrieval regressions. Retrieval is deterministic, so runs are reproducible.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import search
from config import EMBEDDING_MODEL
from eval.metrics import mean_reciprocal_rank, recall_at_k, reciprocal_rank

EVAL_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVAL_DIR / "dataset.json"
RESULTS_DIR = EVAL_DIR / "results"


def _ordered_unique_sources(hits: list[dict]) -> list[str]:
    """Collapse ranked chunk hits to a ranked, de-duplicated document list."""
    seen: list[str] = []
    for hit in hits:
        src = hit["source"]
        if src not in seen:
            seen.append(src)
    return seen


def load_dataset() -> list[dict]:
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if not data:
        raise SystemExit(f"Eval dataset {DATASET_PATH} is empty")
    return data


def evaluate(dataset: list[dict], k: int) -> dict:
    per_query = []
    rankings: list[tuple[list[str], set[str]]] = []

    for case in dataset:
        query = case["query"]
        mode = case.get("mode", "hybrid")
        relevant = set(case["relevant_sources"])

        hits = search.search(query, mode=mode, top_k=max(k, 10))
        retrieved = _ordered_unique_sources(hits)

        r_at_k = recall_at_k(retrieved, relevant, k)
        rr = reciprocal_rank(retrieved, relevant)
        rankings.append((retrieved, relevant))

        per_query.append({
            "id": case.get("id", query[:40]),
            "query": query,
            "mode": mode,
            "recall_at_k": round(r_at_k, 4),
            "reciprocal_rank": round(rr, 4),
            "retrieved_top": retrieved[:k],
            "relevant": sorted(relevant),
        })

    recall = sum(q["recall_at_k"] for q in per_query) / len(per_query)
    mrr = mean_reciprocal_rank(rankings)

    return {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "embedding_model": EMBEDDING_MODEL,
        "k": k,
        "num_queries": len(per_query),
        "recall_at_k": round(recall, 4),
        "mrr": round(mrr, 4),
        "per_query": per_query,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bahith retrieval mini-eval")
    parser.add_argument("--k", type=int, default=5, help="Cutoff for recall@k")
    parser.add_argument("--min-recall", type=float, default=0.0, help="CI gate: min mean recall@k")
    parser.add_argument("--min-mrr", type=float, default=0.0, help="CI gate: min MRR")
    parser.add_argument("--save", action="store_true", help="Write JSON report to eval/results/")
    args = parser.parse_args()

    report = evaluate(load_dataset(), args.k)

    print(f"Bahith retrieval eval  ({report['timestamp']})")
    print(f"  embedding model : {report['embedding_model']}")
    print(f"  queries         : {report['num_queries']}")
    print(f"  recall@{args.k}       : {report['recall_at_k']:.3f}")
    print(f"  MRR             : {report['mrr']:.3f}")

    if args.save:
        RESULTS_DIR.mkdir(exist_ok=True)
        out = RESULTS_DIR / f"eval_{report['timestamp'].replace(':', '')}.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  saved report    : {out}")

    failures = []
    if report["recall_at_k"] < args.min_recall:
        failures.append(f"recall@{args.k} {report['recall_at_k']:.3f} < {args.min_recall}")
    if report["mrr"] < args.min_mrr:
        failures.append(f"MRR {report['mrr']:.3f} < {args.min_mrr}")

    if failures:
        print("\nFAILED thresholds:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
