from typing import List, Dict
import embeddings
import bm25
from preprocessor import preprocess
from config import SEARCH_ALPHA, TOP_K


def normalize_scores(results: List[Dict]) -> List[Dict]:
    if not results:
        return results

    scores = [r['score'] for r in results]
    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        for r in results:
            r['norm_score'] = 1.0
    else:
        for r in results:
            r['norm_score'] = (r['score'] - min_score) / (max_score - min_score)

    return results


def semantic_search(query: str, top_k: int = TOP_K) -> List[Dict]:
    query = preprocess(query)
    results = embeddings.search(query, top_k=top_k * 2)
    return normalize_scores(results)


def keyword_search(query: str, top_k: int = TOP_K) -> List[Dict]:
    results = bm25.search(query, top_k=top_k * 2)
    return normalize_scores(results)


def hybrid_search(query: str, alpha: float = SEARCH_ALPHA, top_k: int = TOP_K) -> List[Dict]:
    sem_results = semantic_search(query, top_k)
    kw_results = keyword_search(query, top_k)

    combined = {}

    for r in sem_results:
        combined[r['id']] = {
            'id': r['id'],
            'content': r['content'],
            'title': r['title'],
            'source': r['source'],
            'semantic_score': r['norm_score'],
            'keyword_score': 0.0
        }

    for r in kw_results:
        if r['id'] in combined:
            combined[r['id']]['keyword_score'] = r['norm_score']
        else:
            combined[r['id']] = {
                'id': r['id'],
                'content': r['content'],
                'title': r['title'],
                'source': r['source'],
                'semantic_score': 0.0,
                'keyword_score': r['norm_score']
            }

    results = []
    for doc in combined.values():
        final_score = alpha * doc['semantic_score'] + (1 - alpha) * doc['keyword_score']
        results.append({
            'id': doc['id'],
            'content': doc['content'],
            'title': doc['title'],
            'source': doc['source'],
            'score': final_score
        })

    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]


def search(query: str, mode: str = "hybrid", top_k: int = TOP_K) -> List[Dict]:
    if mode == "semantic":
        results = semantic_search(query, top_k)
        return [{'id': r['id'], 'content': r['content'], 'title': r['title'],
                 'source': r['source'], 'score': r['norm_score']} for r in results[:top_k]]
    elif mode == "keyword":
        results = keyword_search(query, top_k)
        return [{'id': r['id'], 'content': r['content'], 'title': r['title'],
                 'source': r['source'], 'score': r['norm_score']} for r in results[:top_k]]
    else:
        return hybrid_search(query, top_k=top_k)


if __name__ == "__main__":
    from ingest import process_documents

    if not bm25.is_loaded():
        print("Loading BM25 index...")
        if not bm25.load_index():
            print("Building BM25 index...")
            docs = process_documents()
            bm25.build_index(docs)
            bm25.save_index()

    query = "الذكاء الاصطناعي والتعلم الآلي"

    print(f"Query: {query}")
    print("\n" + "=" * 60)

    print("\n[SEMANTIC SEARCH]")
    results = search(query, mode="semantic", top_k=3)
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['score']:.3f}] {r['title']}: {r['content'][:100]}...")

    print("\n" + "=" * 60)
    print("\n[KEYWORD SEARCH]")
    results = search(query, mode="keyword", top_k=3)
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['score']:.3f}] {r['title']}: {r['content'][:100]}...")

    print("\n" + "=" * 60)
    print("\n[HYBRID SEARCH]")
    results = search(query, mode="hybrid", top_k=3)
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['score']:.3f}] {r['title']}: {r['content'][:100]}...")
