import pickle

from rank_bm25 import BM25Okapi

from config import BM25_INDEX_PATH
from preprocessor import preprocess
from schemas import Document, SearchHit

_bm25_index: BM25Okapi | None = None
_documents: list[Document] | None = None


def tokenize(text: str) -> list[str]:
    return text.split()


def build_index(docs: list[Document]) -> None:
    global _bm25_index, _documents

    _documents = docs
    tokenized = [tokenize(d["content"]) for d in docs]
    _bm25_index = BM25Okapi(tokenized)


def save_index() -> None:
    if _bm25_index is None or _documents is None:
        return
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"index": _bm25_index, "documents": _documents}, f)


def load_index() -> bool:
    global _bm25_index, _documents

    if not BM25_INDEX_PATH.exists():
        return False

    with open(BM25_INDEX_PATH, "rb") as f:
        data = pickle.load(f)
        _bm25_index = data["index"]
        _documents = data["documents"]
    return True


def search(query: str, top_k: int = 5) -> list[SearchHit]:
    global _bm25_index, _documents

    if _bm25_index is None and not load_index():
        return []

    query = preprocess(query)
    tokenized_query = tokenize(query)

    scores = _bm25_index.get_scores(tokenized_query)

    indexed_scores = [(i, score) for i, score in enumerate(scores)]
    indexed_scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    for i, score in indexed_scores[:top_k]:
        doc = _documents[i]
        results.append({
            'id': doc['id'],
            'content': doc['content'],
            'title': doc['title'],
            'source': doc['source'],
            'score': float(score)
        })

    return results


def is_loaded() -> bool:
    return _bm25_index is not None


if __name__ == "__main__":
    from ingest import process_documents

    print("Processing documents...")
    docs = process_documents()
    print(f"Building BM25 index with {len(docs)} chunks...")

    build_index(docs)
    save_index()

    print("\nTesting BM25 search with query: 'الذكاء الاصطناعي'")
    results = search("الذكاء الاصطناعي", top_k=5)
    for i, r in enumerate(results, 1):
        print(f"\n{i}. [{r['score']:.3f}] {r['title']}")
        print(f"   {r['content'][:150]}...")

    print("\n" + "=" * 50)
    print("\nTesting BM25 search with query: 'المملكة العربية السعودية'")
    results = search("المملكة العربية السعودية", top_k=5)
    for i, r in enumerate(results, 1):
        print(f"\n{i}. [{r['score']:.3f}] {r['title']}")
        print(f"   {r['content'][:150]}...")
