from fastapi.testclient import TestClient

import generate
import main
import search


def _fake_hits(*_args, **_kwargs):
    return [
        {
            "id": "ai.txt_0",
            "title": "الذكاء الاصطناعي",
            "content": "الذكاء الاصطناعي فرع من علوم الحاسوب. " * 20,
            "source": "ai.txt",
            "score": 0.91,
        }
    ]


def _fake_answer(*_args, **_kwargs):
    return {"answer": "إجابة تجريبية", "confidence": 8, "related": ["سؤال مرتبط"]}


def test_health_ok():
    with TestClient(main.app) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_search_happy_path(monkeypatch):
    monkeypatch.setattr(search, "search", _fake_hits)
    monkeypatch.setattr(generate, "generate_answer", _fake_answer)

    with TestClient(main.app) as client:
        resp = client.post("/search", json={"query": "ما هو الذكاء الاصطناعي", "mode": "hybrid"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "إجابة تجريبية"
    assert body["confidence"] == 8
    assert body["total_results"] == 1
    # Snippet must be truncated with an ellipsis for long content.
    assert body["results"][0]["snippet"].endswith("...")


def test_empty_query_is_rejected_by_validation():
    with TestClient(main.app) as client:
        assert client.post("/search", json={"query": ""}).status_code == 422


def test_whitespace_query_returns_400():
    with TestClient(main.app) as client:
        assert client.post("/search", json={"query": "   "}).status_code == 400


def test_invalid_mode_is_rejected():
    with TestClient(main.app) as client:
        resp = client.post("/search", json={"query": "بحث", "mode": "telepathy"})
    assert resp.status_code == 422
