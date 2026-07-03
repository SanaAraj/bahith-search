import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import bm25
import embeddings
import generate
import search
import web_search
from config import STATIC_DIR, WARMUP_ON_STARTUP
from ingest import process_documents
from observability import trace_query
from schemas import SearchRequest, SearchResponse, SearchResultModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("bahith")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure the BM25 keyword index is ready before serving.

    The vector store (ChromaDB) is built separately via ``build_index.py`` so
    that the heavy embedding step happens at image-build time, not per request.
    """
    if not bm25.is_loaded() and not bm25.load_index():
        docs = process_documents()
        if docs:
            bm25.build_index(docs)
            bm25.save_index()
            logger.info("Built BM25 index from %d chunks", len(docs))
        else:
            logger.warning(
                "No documents found. Run `python build_index.py` to seed and "
                "index content before querying."
            )

    # Warm the embedding model and vector collection so the first user query
    # doesn't pay the one-time model load. Failures here must not block serving
    # (keyword mode still works without embeddings).
    if WARMUP_ON_STARTUP:
        try:
            embeddings.get_embedding_model()
            embeddings.get_collection()
            logger.info("Embedding model warmed")
        except Exception:
            logger.warning(
                "Embedding warmup failed; first semantic query will be slower", exc_info=True
            )

    yield


app = FastAPI(
    title="Bahith",
    description="Arabic Semantic Search Engine",
    lifespan=lifespan,
)

# The API is stateless and uses no cookies, so a permissive origin policy is
# safe here. Credentials must stay disabled for a wildcard origin to be valid.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/suggest")
async def suggest(q: str = ""):
    q_lower = q.strip()
    if len(q_lower) < 2:
        return {"suggestions": []}

    suggestions: set[str] = set()
    docs = bm25._documents or []

    for doc in docs[:500]:
        title = doc.get("title", "")
        if q_lower in title:
            suggestions.add(title)
        words = doc.get("content", "")[:200].split()
        for i, word in enumerate(words):
            if word.startswith(q_lower) or q_lower in word:
                phrase = " ".join(words[max(0, i - 1):i + 3])
                if 5 < len(phrase) < 60:
                    suggestions.add(phrase)

        if len(suggestions) >= 5:
            break

    return {"suggestions": list(suggestions)[:5]}


@app.post("/search", response_model=SearchResponse)
async def search_endpoint(request: SearchRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    start_time = time.time()

    with trace_query(query, mode=request.mode, top_k=request.top_k) as tr:
        try:
            with tr.span("retrieval", mode=request.mode):
                if request.mode == "web":
                    results = web_search.live_web_search(query, max_results=request.top_k)
                else:
                    results = search.search(query, mode=request.mode, top_k=request.top_k)
        except Exception:
            logger.exception("Search failed for query=%r mode=%s", query, request.mode)
            raise HTTPException(status_code=500, detail="Search failed") from None

        answer = None
        confidence = 0
        related_queries: list[str] = []

        if results:
            with tr.span("generation"):
                gen_result = generate.generate_answer(query, results)
            answer = gen_result.get("answer")
            confidence = gen_result.get("confidence", 0)
            related_queries = gen_result.get("related", [])

        tr.update(output={"num_results": len(results), "confidence": confidence})

    search_results = []
    for r in results:
        content = r["content"]
        snippet = content[:300] + "..." if len(content) > 300 else content
        search_results.append(
            SearchResultModel(
                title=r["title"],
                snippet=snippet,
                source=r["source"],
                score=r["score"],
            )
        )

    return SearchResponse(
        query=query,
        answer=answer,
        confidence=confidence,
        related_queries=related_queries,
        results=search_results,
        total_results=len(search_results),
        search_time=round(time.time() - start_time, 2),
        mode=request.mode,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
