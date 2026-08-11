import math

import pytest

from eval.score import evaluate


def test_perfect_ranking_scores_one():
    result = evaluate({"q1": ["a", "b"]}, {"q1": {"a": 1, "b": 1}}, k=10)
    assert result["ndcg@10"] == pytest.approx(1.0, rel=1e-6)
    assert result["mrr@10"] == pytest.approx(1.0, rel=1e-6)


def test_relevant_doc_in_second_position_halves_mrr():
    result = evaluate({"q1": ["b", "a"]}, {"q1": {"a": 1}}, k=10)
    assert result["mrr@10"] == pytest.approx(0.5, rel=1e-6)


def test_run_is_truncated_at_k():
    result = evaluate({"q1": ["b", "c", "a"]}, {"q1": {"a": 1}}, k=2)
    assert result["mrr@2"] == 0.0
    assert result["ndcg@2"] == 0.0


def test_ndcg_matches_hand_computed_value():
    expected = 1.5 / (1.0 + 1.0 / math.log2(3))
    result = evaluate({"q1": ["a", "b", "c"]}, {"q1": {"a": 1, "c": 1}}, k=3)
    assert result["ndcg@3"] == pytest.approx(expected, rel=1e-6)


def test_graded_relevance_rewards_putting_the_higher_grade_first():
    qrels = {"q1": {"a": 2, "b": 1}}
    better = evaluate({"q1": ["a", "b"]}, qrels, k=10)["ndcg@10"]
    worse = evaluate({"q1": ["b", "a"]}, qrels, k=10)["ndcg@10"]
    assert better > worse


def test_averages_over_queries():
    run = {"q1": ["a", "b"], "q2": ["b", "a"]}
    qrels = {"q1": {"a": 1}, "q2": {"a": 1}}
    result = evaluate(run, qrels, k=10)
    assert result["queries"] == 2
    assert result["mrr@10"] == pytest.approx(0.75, rel=1e-6)


def test_queries_without_judgements_are_ignored():
    result = evaluate({"q1": ["a"], "q_unjudged": ["b"]}, {"q1": {"a": 1}}, k=10)
    assert result["queries"] == 1
    assert result["ndcg@10"] == pytest.approx(1.0, rel=1e-6)


def test_no_judged_queries_returns_zeros():
    assert evaluate({"q1": ["a"]}, {}, k=10) == {
        "ndcg@10": 0.0,
        "mrr@10": 0.0,
        "queries": 0,
    }


def test_empty_ranking_is_skipped():
    result = evaluate({"q1": [], "q2": ["a"]}, {"q1": {"a": 1}, "q2": {"a": 1}}, k=10)
    assert result["queries"] == 1
