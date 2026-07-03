"""Shared data schemas.

Two layers:

* Internal retrieval shapes (:class:`Document`, :class:`SearchHit`) are
  ``TypedDict``s. They document the dict contracts passed between the
  retrieval modules without adding runtime overhead on the hot path.
* Request/response models at the HTTP boundary are pydantic models so that
  FastAPI validates and serialises them.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field

SearchMode = Literal["semantic", "keyword", "hybrid", "web"]


class Document(TypedDict):
    """A chunked document as produced by the ingestion pipeline."""

    id: str
    title: str
    content: str
    source: str
    chunk_index: int


class SearchHit(TypedDict):
    """A single retrieval result flowing through the search pipeline."""

    id: str
    title: str
    content: str
    source: str
    score: float


# --- HTTP boundary models -------------------------------------------------


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    mode: SearchMode = "hybrid"
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResultModel(BaseModel):
    title: str
    snippet: str
    source: str
    score: float


class SearchResponse(BaseModel):
    query: str
    answer: str | None = None
    confidence: int = 0
    related_queries: list[str] = []
    results: list[SearchResultModel]
    total_results: int
    search_time: float
    mode: str = "hybrid"
