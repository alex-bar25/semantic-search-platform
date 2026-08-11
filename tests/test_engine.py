import pytest

from search import engine as engine_module
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
    monkeypatch.setattr(
        engine_module.keyword, "search", lambda conn, q, k: [("k1", 9.0), ("k2", 8.0)]
    )
    monkeypatch.setattr(
        engine_module.vector,
        "search",
        lambda collection, model, q, k: [("v1", 0.9), ("v2", 0.8)],
    )
    return SearchEngine(conn=object(), collection=FakeCollection(DOCS), model=object())


def test_keyword_mode_returns_only_keyword_hits(engine):
    assert [h.doc_id for h in engine.search("q", "keyword", 10).hits] == ["k1", "k2"]


def test_vector_mode_returns_only_vector_hits(engine):
    assert [h.doc_id for h in engine.search("q", "vector", 10).hits] == ["v1", "v2"]


def test_hybrid_mode_fuses_both_retrievers(engine):
    hits = engine.search("q", "hybrid", 10).hits
    assert {h.doc_id for h in hits} == {"k1", "k2", "v1", "v2"}


def test_hybrid_mode_ranks_vector_above_keyword_given_the_tuned_weights(engine):
    assert engine.search("q", "hybrid", 10).hits[0].doc_id == "v1"


def test_hybrid_mode_respects_top_k(engine):
    assert len(engine.search("q", "hybrid", 2).hits) == 2


def test_unknown_mode_raises(engine):
    with pytest.raises(ValueError):
        engine.search("q", "magic", 10)


def test_rerank_without_a_trained_model_raises(engine):
    with pytest.raises(RuntimeError):
        engine.search("q", "rerank", 10)


def test_keyword_hits_carry_their_keyword_rank(engine):
    hits = engine.search("q", "keyword", 10).hits
    assert [h.keyword_rank for h in hits] == [1, 2]
    assert all(h.vector_rank is None for h in hits)


def test_hybrid_hits_explain_both_source_ranks(engine):
    by_id = {h.doc_id: h for h in engine.search("q", "hybrid", 10).hits}
    assert by_id["k1"].keyword_rank == 1
    assert by_id["k1"].vector_rank is None
    assert by_id["v1"].vector_rank == 1
    assert by_id["v1"].keyword_rank is None
    assert by_id["v1"].rrf_score > 0


def test_keyword_mode_reports_only_keyword_timing(engine):
    assert list(engine.search("q", "keyword", 10).timings) == ["keyword_ms"]


def test_hybrid_mode_reports_a_timing_per_stage(engine):
    timings = engine.search("q", "hybrid", 10).timings
    assert set(timings) == {"keyword_ms", "vector_ms", "fusion_ms"}
    assert all(v >= 0 for v in timings.values())


def test_rerank_mode_reports_rerank_timing_and_scores(monkeypatch):
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
    outcome = engine.search("q", "rerank", 10)
    assert [h.doc_id for h in outcome.hits] == ["k1"]
    assert outcome.hits[0].rerank_score == 5.0
    assert outcome.hits[0].rrf_score > 0
    assert "rerank_ms" in outcome.timings


def test_repeated_query_is_served_from_cache(engine):
    before = engine.cache_hits()
    engine.search("same query", "hybrid", 10)
    assert engine.cache_hits() == before
    engine.search("same query", "hybrid", 10)
    assert engine.cache_hits() == before + 1


def test_cache_is_keyed_on_mode_and_top_k(engine):
    engine.search("q", "hybrid", 10)
    before = engine.cache_hits()
    engine.search("q", "hybrid", 5)
    engine.search("q", "keyword", 10)
    assert engine.cache_hits() == before


def test_texts_for_preserves_requested_order(engine):
    assert engine.texts_for(["v1", "k1"]) == ["vector one", "keyword one"]


def test_texts_for_returns_empty_string_for_missing_docs(engine):
    assert engine.texts_for(["nope"]) == [""]


def test_texts_for_with_no_ids_returns_empty(engine):
    assert engine.texts_for([]) == []
