"""Retrieval quality metrics used by the mini-evaluation suite.

Both metrics operate on a ranked list of retrieved document identifiers and a
set of identifiers judged relevant for the query.
"""

from __future__ import annotations

from collections.abc import Sequence


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant documents that appear in the top-``k`` results.

    Returns 0.0 when there are no relevant documents (an undefined query is
    treated as a miss rather than a division error).
    """
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    hit = len(top_k & relevant)
    return hit / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """Reciprocal of the rank of the first relevant hit, or 0.0 if none."""
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(rankings: list[tuple[Sequence[str], set[str]]]) -> float:
    """MRR across queries; 0.0 for an empty set of queries."""
    if not rankings:
        return 0.0
    return sum(reciprocal_rank(r, rel) for r, rel in rankings) / len(rankings)
