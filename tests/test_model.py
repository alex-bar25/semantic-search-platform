import pytest

from rerank.model import load_trained, new_model, rerank


@pytest.fixture(scope="module")
def model():
    return new_model()


def test_rerank_returns_doc_id_score_pairs(model):
    doc_ids = ["d1", "d2"]
    texts = ["an etf is an exchange traded fund", "the weather is cold today"]
    results = rerank(model, "what is an etf", doc_ids, texts)
    assert len(results) == 2
    assert {doc_id for doc_id, _ in results} == {"d1", "d2"}
    assert all(isinstance(score, float) for _, score in results)


def test_rerank_sorts_by_descending_score(model):
    results = rerank(
        model,
        "what is an etf",
        ["d1", "d2", "d3"],
        [
            "an etf is an exchange traded fund holding many assets",
            "the weather is cold today",
            "a mutual fund pools money from many investors",
        ],
    )
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_rerank_puts_the_relevant_passage_first(model):
    results = rerank(
        model,
        "what is an exchange traded fund",
        ["irrelevant", "relevant"],
        [
            "my dog likes going for walks in the park",
            "an exchange traded fund is a basket of securities traded on an exchange",
        ],
    )
    assert results[0][0] == "relevant"


def test_rerank_with_no_candidates_returns_empty(model):
    assert rerank(model, "query", [], []) == []


def test_load_trained_without_checkpoint_raises():
    with pytest.raises(FileNotFoundError):
        load_trained("/nonexistent/path/checkpoint")
