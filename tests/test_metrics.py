from eval.metrics import (
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_counts_hits_in_top_k():
    retrieved = ["a", "b", "c", "d", "e"]
    assert recall_at_k(retrieved, {"a", "c"}, k=5) == 1.0
    assert recall_at_k(retrieved, {"a", "z"}, k=5) == 0.5


def test_recall_at_k_respects_cutoff():
    retrieved = ["x", "y", "a"]
    assert recall_at_k(retrieved, {"a"}, k=2) == 0.0
    assert recall_at_k(retrieved, {"a"}, k=3) == 1.0


def test_recall_at_k_no_relevant_is_zero():
    assert recall_at_k(["a", "b"], set(), k=5) == 0.0


def test_reciprocal_rank_uses_first_hit_position():
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert reciprocal_rank(["a", "b", "c"], {"a", "c"}) == 1.0
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0


def test_mrr_averages_reciprocal_ranks():
    rankings = [
        (["a", "b"], {"a"}),  # rr = 1.0
        (["x", "y", "z"], {"z"}),  # rr = 1/3
    ]
    assert mean_reciprocal_rank(rankings) == (1.0 + 1 / 3) / 2


def test_mrr_empty_is_zero():
    assert mean_reciprocal_rank([]) == 0.0
