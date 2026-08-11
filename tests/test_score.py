import math

import pytest

from eval.score import dcg, mrr_at_k, ndcg_at_k


def test_dcg_hand_computed():
    assert dcg([1, 0, 1], k=3) == 1.5


def test_dcg_truncates_at_k():
    assert dcg([1, 0, 1], k=2) == 1.0


def test_ndcg_hand_computed():
    ranked = ["a", "b", "c"]
    relevant = {"a": 1, "c": 1}
    expected = 1.5 / (1.0 + 1.0 / math.log2(3))
    assert ndcg_at_k(ranked, relevant, k=3) == pytest.approx(expected, rel=1e-9)


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b"], {"a": 1, "b": 1}, k=2) == pytest.approx(1.0, rel=1e-9)


def test_ndcg_no_relevant_docs_is_zero():
    assert ndcg_at_k(["a", "b"], {}, k=2) == 0.0


def test_ndcg_respects_graded_relevance():
    better = ndcg_at_k(["a", "b"], {"a": 2, "b": 1}, k=2)
    worse = ndcg_at_k(["b", "a"], {"a": 2, "b": 1}, k=2)
    assert better > worse


def test_ndcg_idcg_uses_only_top_k_ideal_gains():
    ranked = ["a"]
    relevant = {"a": 1, "b": 1, "c": 1}
    assert ndcg_at_k(ranked, relevant, k=1) == pytest.approx(1.0, rel=1e-9)


def test_mrr_first_position():
    assert mrr_at_k(["a", "b", "c"], {"a": 1}, k=10) == 1.0


def test_mrr_second_position():
    assert mrr_at_k(["b", "a", "c"], {"a": 1}, k=10) == 0.5


def test_mrr_outside_k_is_zero():
    assert mrr_at_k(["b", "c", "a"], {"a": 1}, k=2) == 0.0


def test_mrr_no_relevant_docs_is_zero():
    assert mrr_at_k(["a", "b"], {}, k=10) == 0.0


def test_evaluate_averages_over_queries():
    from eval.score import evaluate

    run = {"q1": ["a", "b"], "q2": ["b", "a"]}
    qrels = {"q1": {"a": 1}, "q2": {"a": 1}}
    result = evaluate(run, qrels, k=10)
    assert result["queries"] == 2
    assert result["mrr@10"] == pytest.approx(0.75, rel=1e-9)


def test_evaluate_ignores_queries_without_judgements():
    from eval.score import evaluate

    run = {"q1": ["a"], "q_unjudged": ["b"]}
    qrels = {"q1": {"a": 1}}
    result = evaluate(run, qrels, k=10)
    assert result["queries"] == 1
    assert result["ndcg@10"] == pytest.approx(1.0, rel=1e-9)


def test_evaluate_with_no_judged_queries_returns_zeros():
    from eval.score import evaluate

    result = evaluate({"q1": ["a"]}, {}, k=10)
    assert result == {"ndcg@10": 0.0, "mrr@10": 0.0, "queries": 0}
