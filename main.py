import time
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import search
import bm25
import generate
import web_search
from ingest import process_documents
from config import TOP_K

app = FastAPI(title="Bahith", description="Arabic Semantic Search Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


class SearchRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    top_k: int = TOP_K


class SearchResult(BaseModel):
    title: str
    snippet: str
    source: str
    score: float


class SearchResponse(BaseModel):
    query: str
    answer: Optional[str]
    results: list[SearchResult]
    total_results: int
    search_time: float


@app.on_event("startup")
async def startup():
    if not bm25.is_loaded():
        if not bm25.load_index():
            docs = process_documents()
            bm25.build_index(docs)
            bm25.save_index()


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/suggest")
async def suggest(q: str = ""):
    if not q or len(q) < 2:
        return {"suggestions": []}

    suggestions = set()
    docs = bm25._documents or []

    q_lower = q.strip()
    for doc in docs[:500]:
        title = doc.get('title', '')
        if q_lower in title:
            suggestions.add(title)
        content = doc.get('content', '')[:200]
        words = content.split()
        for i, word in enumerate(words):
            if word.startswith(q_lower) or q_lower in word:
                phrase = ' '.join(words[max(0, i-1):i+3])
                if len(phrase) > 5 and len(phrase) < 60:
                    suggestions.add(phrase)

        if len(suggestions) >= 5:
            break

    return {"suggestions": list(suggestions)[:5]}


@app.post("/search", response_model=SearchResponse)
async def search_endpoint(request: SearchRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    query = request.query.strip()
    if len(query) > 500:
        query = query[:500]

    mode = request.mode if request.mode in ("semantic", "keyword", "hybrid", "web") else "hybrid"
    top_k = min(max(request.top_k, 1), 20)

    start_time = time.time()

    try:
        if mode == "web":
            results = web_search.live_web_search(query, max_results=top_k)
        else:
            results = search.search(query, mode=mode, top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Search error")

    answer = None
    if results:
        try:
            answer = generate.generate_answer(query, results)
        except:
            pass

    search_results = []
    for r in results:
        snippet = r['content'][:300] + "..." if len(r['content']) > 300 else r['content']
        search_results.append(SearchResult(
            title=r['title'],
            snippet=snippet,
            source=r['source'],
            score=r['score']
        ))

    search_time = time.time() - start_time

    return SearchResponse(
        query=query,
        answer=answer,
        results=search_results,
        total_results=len(search_results),
        search_time=round(search_time, 2)
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
