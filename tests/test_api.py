import pytest
from fastapi.testclient import TestClient

from api.app import app, get_engine

TEXTS = {"d1": "text of d1", "d2": "text of d2"}


class FakeEngine:
    def search(self, query, mode="hybrid", top_k=10):
        if mode == "rerank":
            raise RuntimeError("No trained re-ranker loaded.")
        return [("d1", 0.9), ("d2", 0.5)][:top_k]

    def texts_for(self, doc_ids):
        return [TEXTS[doc_id] for doc_id in doc_ids]


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
    response = client.post("/search", json={"query": "etf", "mode": "magic"})
    assert response.status_code == 422


def test_top_k_above_limit_is_rejected(client):
    response = client.post("/search", json={"query": "etf", "top_k": 1000})
    assert response.status_code == 422


def test_missing_reranker_returns_503(client):
    response = client.post("/search", json={"query": "etf", "mode": "rerank"})
    assert response.status_code == 503
    assert "re-ranker" in response.json()["detail"]
