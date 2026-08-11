import time
from dataclasses import dataclass, field
from functools import lru_cache

from index import keyword, vector
from rerank.model import rerank

CANDIDATE_K = 100
RERANK_K = 50
HYBRID_WEIGHTS = [0.1, 1.0]
CACHE_SIZE = 256
MODES = ("keyword", "vector", "hybrid", "rerank")


@dataclass
class Hit:
    doc_id: str
    score: float
    keyword_rank: int | None = None
    vector_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


@dataclass
class SearchOutcome:
    hits: list[Hit] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)


def rrf(
    rankings: list[list[str]], k: int = 60, weights: list[float] | None = None
) -> list[tuple[str, float]]:
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(f"got {len(weights)} weights for {len(rankings)} rankings")
    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda pair: -pair[1])


def _ranks(rows: list[tuple[str, float]]) -> dict[str, int]:
    return {doc_id: i + 1 for i, (doc_id, _) in enumerate(rows)}


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


class SearchEngine:
    def __init__(self, conn=None, collection=None, model=None, reranker=None):
        self.conn = conn if conn is not None else keyword.connect()
        self.collection = collection if collection is not None else vector.connect()
        self.model = model if model is not None else vector.get_model()
        self.reranker = reranker
        self._cached_search = lru_cache(maxsize=CACHE_SIZE)(self._search)

    def search(
        self, query: str, mode: str = "hybrid", top_k: int = 10
    ) -> SearchOutcome:
        return self._cached_search(query, mode, top_k)

    def cache_hits(self) -> int:
        return self._cached_search.cache_info().hits

    def _search(self, query: str, mode: str, top_k: int) -> SearchOutcome:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

        timings: dict[str, float] = {}

        if mode == "keyword":
            started = time.perf_counter()
            rows = keyword.search(self.conn, query, top_k)
            timings["keyword_ms"] = _elapsed_ms(started)
            hits = [
                Hit(doc_id=doc_id, score=score, keyword_rank=i + 1)
                for i, (doc_id, score) in enumerate(rows)
            ]
            return SearchOutcome(hits, timings)

        if mode == "vector":
            started = time.perf_counter()
            rows = vector.search(self.collection, self.model, query, top_k)
            timings["vector_ms"] = _elapsed_ms(started)
            hits = [
                Hit(doc_id=doc_id, score=score, vector_rank=i + 1)
                for i, (doc_id, score) in enumerate(rows)
            ]
            return SearchOutcome(hits, timings)

        started = time.perf_counter()
        keyword_rows = keyword.search(self.conn, query, CANDIDATE_K)
        timings["keyword_ms"] = _elapsed_ms(started)

        started = time.perf_counter()
        vector_rows = vector.search(self.collection, self.model, query, CANDIDATE_K)
        timings["vector_ms"] = _elapsed_ms(started)

        started = time.perf_counter()
        fused = rrf(
            [[d for d, _ in keyword_rows], [d for d, _ in vector_rows]],
            weights=HYBRID_WEIGHTS,
        )
        timings["fusion_ms"] = _elapsed_ms(started)

        keyword_ranks = _ranks(keyword_rows)
        vector_ranks = _ranks(vector_rows)
        rrf_scores = dict(fused)

        if mode == "hybrid":
            hits = [
                Hit(
                    doc_id=doc_id,
                    score=score,
                    rrf_score=score,
                    keyword_rank=keyword_ranks.get(doc_id),
                    vector_rank=vector_ranks.get(doc_id),
                )
                for doc_id, score in fused[:top_k]
            ]
            return SearchOutcome(hits, timings)

        if self.reranker is None:
            raise RuntimeError(
                "No trained re-ranker loaded. Train one with: uv run python -m rerank.train"
            )

        candidate_ids = [doc_id for doc_id, _ in fused[:RERANK_K]]
        texts = self.texts_for(candidate_ids)
        started = time.perf_counter()
        ranked = rerank(self.reranker, query, candidate_ids, texts)
        timings["rerank_ms"] = _elapsed_ms(started)

        hits = [
            Hit(
                doc_id=doc_id,
                score=score,
                rerank_score=score,
                rrf_score=rrf_scores.get(doc_id),
                keyword_rank=keyword_ranks.get(doc_id),
                vector_rank=vector_ranks.get(doc_id),
            )
            for doc_id, score in ranked[:top_k]
        ]
        return SearchOutcome(hits, timings)

    def texts_for(self, doc_ids: list[str]) -> list[str]:
        if not doc_ids:
            return []
        result = self.collection.get(ids=doc_ids)
        by_id = dict(zip(result["ids"], result["documents"]))
        return [by_id.get(doc_id, "") for doc_id in doc_ids]
