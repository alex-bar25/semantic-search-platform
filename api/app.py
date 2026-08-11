import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from rerank.model import is_trained, load_trained
from search.engine import MODES, SearchEngine, SearchOutcome


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = SearchEngine(reranker=load_trained() if is_trained() else None)
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


class CompareRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=50)


class SearchResult(BaseModel):
    doc_id: str
    text: str
    score: float
    rank: int
    keyword_rank: int | None = None
    vector_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


class SearchResponse(BaseModel):
    query: str
    mode: str
    results: list[SearchResult]
    timings_ms: dict[str, float]
    took_ms: float
    cached: bool


class CompareResponse(BaseModel):
    query: str
    modes: dict[str, list[SearchResult]]
    timings_ms: dict[str, dict[str, float]]
    took_ms: float


def to_results(engine: SearchEngine, outcome: SearchOutcome) -> list[SearchResult]:
    texts = engine.texts_for([hit.doc_id for hit in outcome.hits])
    return [
        SearchResult(
            doc_id=hit.doc_id,
            text=text,
            score=hit.score,
            rank=i + 1,
            keyword_rank=hit.keyword_rank,
            vector_rank=hit.vector_rank,
            rrf_score=hit.rrf_score,
            rerank_score=hit.rerank_score,
        )
        for i, (hit, text) in enumerate(zip(outcome.hits, texts))
    ]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest, engine: SearchEngine = Depends(get_engine)):
    started = time.perf_counter()
    hits_before = engine.cache_hits()
    try:
        outcome = engine.search(request.query, request.mode, request.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return SearchResponse(
        query=request.query,
        mode=request.mode,
        results=to_results(engine, outcome),
        timings_ms=outcome.timings,
        took_ms=round((time.perf_counter() - started) * 1000, 2),
        cached=engine.cache_hits() > hits_before,
    )


@app.post("/compare", response_model=CompareResponse)
def compare(request: CompareRequest, engine: SearchEngine = Depends(get_engine)):
    started = time.perf_counter()
    modes: dict[str, list[SearchResult]] = {}
    timings: dict[str, dict[str, float]] = {}

    for mode in MODES:
        try:
            outcome = engine.search(request.query, mode, request.top_k)
        except RuntimeError:
            continue
        modes[mode] = to_results(engine, outcome)
        timings[mode] = outcome.timings

    return CompareResponse(
        query=request.query,
        modes=modes,
        timings_ms=timings,
        took_ms=round((time.perf_counter() - started) * 1000, 2),
    )
