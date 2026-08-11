import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from rerank.model import is_trained, load_trained
from search.engine import SearchEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = SearchEngine(
        reranker=load_trained() if is_trained() else None
    )
    yield
    app.state.engine = None


app = FastAPI(title="Semantic Search Platform", lifespan=lifespan)


def get_engine(request: Request) -> SearchEngine:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Search engine not ready")
    return engine


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=100)
    mode: Literal["keyword", "vector", "hybrid", "rerank"] = "hybrid"


class SearchResult(BaseModel):
    doc_id: str
    text: str
    score: float
    rank: int


class SearchResponse(BaseModel):
    query: str
    mode: str
    results: list[SearchResult]
    took_ms: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest, engine: SearchEngine = Depends(get_engine)):
    started = time.perf_counter()
    try:
        hits = engine.search(request.query, request.mode, request.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    texts = engine.texts_for([doc_id for doc_id, _ in hits])
    results = [
        SearchResult(doc_id=doc_id, text=text, score=score, rank=i + 1)
        for i, ((doc_id, score), text) in enumerate(zip(hits, texts))
    ]
    return SearchResponse(
        query=request.query,
        mode=request.mode,
        results=results,
        took_ms=(time.perf_counter() - started) * 1000,
    )
