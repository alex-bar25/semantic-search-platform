import pytest

from search.engine import SearchEngine


class FakeCollection:
    def __init__(self, docs: dict[str, str]):
        self.docs = docs

    def get(self, ids):
        present = [doc_id for doc_id in ids if doc_id in self.docs]
        return {"ids": present, "documents": [self.docs[i] for i in present]}


DOCS = {"k1": "keyword one", "k2": "keyword two", "v1": "vector one", "v2": "vector two"}


@pytest.fixture
def engine(monkeypatch):
    from search import engine as engine_module

    monkeypatch.setattr(
        engine_module.keyword, "search", lambda conn, q, k: [("k1", 9.0), ("k2", 8.0)]
    )
    monkeypatch.setattr(
        engine_module.vector,
        "search",
        lambda collection, model, q, k: [("v1", 0.9), ("v2", 0.8)],
    )
    return SearchEngine(
        conn=object(), collection=FakeCollection(DOCS), model=object()
    )


def test_keyword_mode_returns_only_keyword_hits(engine):
    assert [d for d, _ in engine.search("q", "keyword", 10)] == ["k1", "k2"]


def test_vector_mode_returns_only_vector_hits(engine):
    assert [d for d, _ in engine.search("q", "vector", 10)] == ["v1", "v2"]


def test_hybrid_mode_fuses_both_retrievers(engine):
    doc_ids = [d for d, _ in engine.search("q", "hybrid", 10)]
    assert set(doc_ids) == {"k1", "k2", "v1", "v2"}


def test_hybrid_mode_ranks_vector_above_keyword_given_the_tuned_weights(engine):
    doc_ids = [d for d, _ in engine.search("q", "hybrid", 10)]
    assert doc_ids[0] == "v1"


def test_hybrid_mode_respects_top_k(engine):
    assert len(engine.search("q", "hybrid", 2)) == 2


def test_unknown_mode_raises(engine):
    with pytest.raises(ValueError):
        engine.search("q", "magic", 10)


def test_rerank_without_a_trained_model_raises(engine):
    with pytest.raises(RuntimeError):
        engine.search("q", "rerank", 10)


def test_rerank_mode_uses_the_reranker(monkeypatch):
    from search import engine as engine_module

    monkeypatch.setattr(
        engine_module.keyword, "search", lambda conn, q, k: [("k1", 9.0)]
    )
    monkeypatch.setattr(
        engine_module.vector, "search", lambda collection, model, q, k: [("v1", 0.9)]
    )
    monkeypatch.setattr(
        engine_module, "rerank", lambda model, query, doc_ids, texts: [("k1", 5.0)]
    )
    engine = SearchEngine(
        conn=object(),
        collection=FakeCollection(DOCS),
        model=object(),
        reranker=object(),
    )
    assert engine.search("q", "rerank", 10) == [("k1", 5.0)]


def test_texts_for_preserves_requested_order(engine):
    assert engine.texts_for(["v1", "k1"]) == ["vector one", "keyword one"]


def test_texts_for_returns_empty_string_for_missing_docs(engine):
    assert engine.texts_for(["nope"]) == [""]


def test_texts_for_with_no_ids_returns_empty(engine):
    assert engine.texts_for([]) == []
