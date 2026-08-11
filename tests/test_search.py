import pytest

from search.engine import rrf


def test_rrf_hand_computed_order():
    rankings = [["a", "b", "c"], ["b", "c", "a"]]
    assert [doc_id for doc_id, _ in rrf(rankings, k=60)] == ["b", "a", "c"]


def test_rrf_hand_computed_score():
    result = dict(rrf([["a", "b", "c"], ["b", "c", "a"]], k=60))
    assert result["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_rewards_appearing_in_both_rankings():
    result = dict(rrf([["solo", "shared"], ["other", "shared"]], k=60))
    assert result["shared"] > result["solo"]


def test_rrf_handles_empty_ranking():
    assert rrf([[], ["a"]], k=60) == [("a", pytest.approx(1 / 61))]


def test_rrf_with_no_rankings_returns_empty():
    assert rrf([], k=60) == []


def test_rrf_weights_scale_each_ranking():
    unweighted = dict(rrf([["a"], ["b"]], k=60))
    assert unweighted["a"] == pytest.approx(unweighted["b"])
    weighted = dict(rrf([["a"], ["b"]], k=60, weights=[2.0, 1.0]))
    assert weighted["a"] == pytest.approx(2 * weighted["b"])


def test_rrf_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        rrf([["a"], ["b"]], weights=[1.0])


def test_rrf_smaller_k_steepens_the_rank_discount():
    flat = dict(rrf([["a", "b"]], k=60))
    sharp = dict(rrf([["a", "b"]], k=1))
    assert sharp["a"] / sharp["b"] > flat["a"] / flat["b"]


