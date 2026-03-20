import os
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from config import EMBEDDING_MODEL, FALLBACK_EMBEDDING_MODEL, CHROMA_PATH

_model = None
_chroma_client = None
_collection = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is not None:
        return _model

    try:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    except Exception:
        _model = SentenceTransformer(FALLBACK_EMBEDDING_MODEL)

    return _model


def get_collection():
    global _chroma_client, _collection
    if _collection is not None:
        return _collection

    _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    _collection = _chroma_client.get_or_create_collection(
        name="arabic_docs",
        metadata={"hnsw:space": "cosine"}
    )
    return _collection


def add_documents(docs: List[Dict]):
    model = get_embedding_model()
    collection = get_collection()

    batch_size = 100
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]

        ids = [d['id'] for d in batch]
        texts = [d['content'] for d in batch]
        metadatas = [{'title': d['title'], 'source': d['source'], 'chunk_index': d['chunk_index']} for d in batch]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )


def search(query: str, top_k: int = 5) -> List[Dict]:
    model = get_embedding_model()
    collection = get_collection()

    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=['documents', 'metadatas', 'distances']
    )

    output = []
    for i in range(len(results['ids'][0])):
        distance = results['distances'][0][i]
        score = 1 - distance

        output.append({
            'id': results['ids'][0][i],
            'content': results['documents'][0][i],
            'title': results['metadatas'][0][i]['title'],
            'source': results['metadatas'][0][i]['source'],
            'score': score
        })

    return output


def get_document_count() -> int:
    collection = get_collection()
    return collection.count()


def clear_collection():
    global _collection
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection("arabic_docs")
    except:
        pass
    _collection = None


if __name__ == "__main__":
    from ingest import process_documents

    print("Loading embedding model...")
    model = get_embedding_model()
    print(f"Model loaded: {model.get_sentence_embedding_dimension()} dimensions")

    print("\nClearing existing collection...")
    clear_collection()

    print("\nProcessing and embedding documents...")
    docs = process_documents()
    print(f"Adding {len(docs)} chunks to vector store...")
    add_documents(docs)

    count = get_document_count()
    print(f"\nTotal embeddings stored: {count}")

    print("\nTesting search with query: 'الذكاء الاصطناعي'")
    results = search("الذكاء الاصطناعي", top_k=5)
    for i, r in enumerate(results, 1):
        print(f"\n{i}. [{r['score']:.3f}] {r['title']}")
        print(f"   {r['content'][:150]}...")
