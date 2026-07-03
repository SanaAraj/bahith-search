import bm25
import embeddings
import search
from search import hybrid_search, normalize_scores


def _hit(doc_id: str, score: float) -> dict:
    return {
        "id": doc_id,
        "title": doc_id.upper(),
        "content": f"content of {doc_id}",
        "source": f"{doc_id}.txt",
        "score": score,
    }


def test_normalize_scores_empty_returns_empty():
    assert normalize_scores([]) == []


def test_normalize_scores_scales_to_unit_range():
    out = normalize_scores([_hit("a", 2.0), _hit("b", 4.0), _hit("c", 6.0)])
    norms = {r["id"]: r["norm_score"] for r in out}
    assert norms["a"] == 0.0
    assert norms["c"] == 1.0
    assert norms["b"] == 0.5


def test_normalize_scores_identical_scores_all_one():
    out = normalize_scores([_hit("a", 5.0), _hit("b", 5.0)])
    assert all(r["norm_score"] == 1.0 for r in out)


def test_hybrid_search_fuses_semantic_and_keyword(monkeypatch):
    # a: strong semantic only; b: strong keyword only; c: weak on both.
    monkeypatch.setattr(embeddings, "search", lambda q, top_k: [_hit("a", 1.0), _hit("b", 0.0)])
    monkeypatch.setattr(bm25, "search", lambda q, top_k: [_hit("b", 10.0), _hit("c", 0.0)])

    results = hybrid_search("الذكاء الاصطناعي", alpha=0.7, top_k=3)

    scores = {r["id"]: round(r["score"], 3) for r in results}
    assert scores == {"a": 0.7, "b": 0.3, "c": 0.0}
    # Ranking must follow the fused score.
    assert [r["id"] for r in results] == ["a", "b", "c"]


def test_hybrid_search_respects_top_k(monkeypatch):
    monkeypatch.setattr(
        embeddings, "search", lambda q, top_k: [_hit(x, 1.0) for x in ("a", "b", "c", "d")]
    )
    monkeypatch.setattr(bm25, "search", lambda q, top_k: [])
    assert len(hybrid_search("test", top_k=2)) == 2


def test_search_dispatch_semantic_returns_clean_hits(monkeypatch):
    monkeypatch.setattr(embeddings, "search", lambda q, top_k: [_hit("a", 0.9)])
    out = search.search("q", mode="semantic", top_k=1)
    assert set(out[0].keys()) == {"id", "title", "content", "source", "score"}
