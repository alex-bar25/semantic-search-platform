from index import keyword, vector
from rerank.model import rerank

CANDIDATE_K = 100
RERANK_K = 50
HYBRID_WEIGHTS = [0.1, 1.0]
MODES = ("keyword", "vector", "hybrid", "rerank")


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


class SearchEngine:
    def __init__(self, conn=None, collection=None, model=None, reranker=None):
        self.conn = conn if conn is not None else keyword.connect()
        self.collection = collection if collection is not None else vector.connect()
        self.model = model if model is not None else vector.get_model()
        self.reranker = reranker

    def search(
        self, query: str, mode: str = "hybrid", top_k: int = 10
    ) -> list[tuple[str, float]]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

        if mode == "keyword":
            return keyword.search(self.conn, query, top_k)
        if mode == "vector":
            return vector.search(self.collection, self.model, query, top_k)

        keyword_hits = [d for d, _ in keyword.search(self.conn, query, CANDIDATE_K)]
        vector_hits = [
            d for d, _ in vector.search(self.collection, self.model, query, CANDIDATE_K)
        ]
        fused = rrf([keyword_hits, vector_hits], weights=HYBRID_WEIGHTS)

        if mode == "hybrid":
            return fused[:top_k]

        if self.reranker is None:
            raise RuntimeError(
                "No trained re-ranker loaded. Train one with: uv run python -m rerank.train"
            )
        candidate_ids = [doc_id for doc_id, _ in fused[:RERANK_K]]
        texts = self.texts_for(candidate_ids)
        return rerank(self.reranker, query, candidate_ids, texts)[:top_k]

    def texts_for(self, doc_ids: list[str]) -> list[str]:
        if not doc_ids:
            return []
        result = self.collection.get(ids=doc_ids)
        by_id = dict(zip(result["ids"], result["documents"]))
        return [by_id.get(doc_id, "") for doc_id in doc_ids]
