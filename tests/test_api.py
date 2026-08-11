import pytest
from fastapi.testclient import TestClient

from api.app import app, get_engine
from search.engine import Hit, SearchOutcome

TEXTS = {"d1": "text of d1", "d2": "text of d2"}


class FakeEngine:
    def __init__(self):
        self.hits = 0

    def search(self, query, mode="hybrid", top_k=10):
        if mode == "rerank":
            raise RuntimeError("No trained re-ranker loaded.")
        hits = [
            Hit(doc_id="d1", score=0.9, keyword_rank=3, vector_rank=1, rrf_score=0.031),
            Hit(doc_id="d2", score=0.5, keyword_rank=1, vector_rank=None, rrf_score=0.016),
        ][:top_k]
        return SearchOutcome(hits, {"keyword_ms": 1.5, "vector_ms": 8.0})

    def texts_for(self, doc_ids):
        return [TEXTS[doc_id] for doc_id in doc_ids]

    def cache_hits(self):
        return self.hits


@pytest.fixture
def client():
    app.dependency_overrides[get_engine] = lambda: FakeEngine()
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_search_returns_ranked_results(client):
    body = client.post("/search", json={"query": "what is an etf"}).json()
    assert [r["doc_id"] for r in body["results"]] == ["d1", "d2"]
    assert [r["rank"] for r in body["results"]] == [1, 2]
    assert [r["text"] for r in body["results"]] == ["text of d1", "text of d2"]


def test_results_carry_explain_fields(client):
    first = client.post("/search", json={"query": "etf"}).json()["results"][0]
    assert first["keyword_rank"] == 3
    assert first["vector_rank"] == 1
    assert first["rrf_score"] == 0.031
    assert first["rerank_score"] is None


def test_response_reports_per_stage_timings(client):
    body = client.post("/search", json={"query": "etf"}).json()
    assert body["timings_ms"] == {"keyword_ms": 1.5, "vector_ms": 8.0}
    assert body["took_ms"] >= 0


def test_search_echoes_query_and_mode(client):
    body = client.post("/search", json={"query": "etf", "mode": "vector"}).json()
    assert body["query"] == "etf"
    assert body["mode"] == "vector"


def test_search_defaults_to_hybrid(client):
    assert client.post("/search", json={"query": "etf"}).json()["mode"] == "hybrid"


def test_search_respects_top_k(client):
    body = client.post("/search", json={"query": "etf", "top_k": 1}).json()
    assert len(body["results"]) == 1


def test_empty_query_is_rejected(client):
    assert client.post("/search", json={"query": ""}).status_code == 422


def test_missing_query_is_rejected(client):
    assert client.post("/search", json={}).status_code == 422


def test_unknown_mode_is_rejected(client):
    assert client.post("/search", json={"query": "e", "mode": "magic"}).status_code == 422


def test_top_k_above_limit_is_rejected(client):
    assert client.post("/search", json={"query": "e", "top_k": 1000}).status_code == 422


def test_missing_reranker_returns_503(client):
    response = client.post("/search", json={"query": "etf", "mode": "rerank"})
    assert response.status_code == 503
    assert "re-ranker" in response.json()["detail"]


def test_compare_returns_every_available_mode(client):
    body = client.post("/compare", json={"query": "etf"}).json()
    assert set(body["modes"]) == {"keyword", "vector", "hybrid"}
    assert set(body["timings_ms"]) == {"keyword", "vector", "hybrid"}


def test_compare_skips_modes_that_are_unavailable(client):
    body = client.post("/compare", json={"query": "etf"}).json()
    assert "rerank" not in body["modes"]


def test_compare_respects_top_k(client):
    body = client.post("/compare", json={"query": "etf", "top_k": 1}).json()
    assert all(len(results) == 1 for results in body["modes"].values())


def test_cached_flag_is_false_on_a_cache_miss(client):
    assert client.post("/search", json={"query": "etf"}).json()["cached"] is False
