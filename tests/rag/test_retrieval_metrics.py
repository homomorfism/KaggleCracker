"""The five metrics against hand-computed values — a scorer nobody tested is
a number nobody should trust."""

import math

import pytest

from evals.retrieval_metrics import (
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

RETRIEVED = ["a", "b", "c", "d", "e"]


def test_hand_computed_values():
    golden = {"b", "e", "zz"}  # zz was never retrieved
    assert hit_rate_at_k(RETRIEVED, golden, 3) == 1.0
    assert hit_rate_at_k(RETRIEVED, {"zz"}, 5) == 0.0
    assert precision_at_k(RETRIEVED, golden, 3) == pytest.approx(1 / 3)
    assert recall_at_k(RETRIEVED, golden, 3) == pytest.approx(1 / 3)
    assert recall_at_k(RETRIEVED, golden, 5) == pytest.approx(2 / 3)
    assert mrr(RETRIEVED, golden) == pytest.approx(1 / 2)  # first hit at rank 2
    # nDCG@3 by hand: DCG = 1/log2(3) (hit at position 2, 0-indexed 1);
    # IDCG for min(3 golden, 3) = 1/log2(2)+1/log2(3)+1/log2(4).
    dcg = 1 / math.log2(3)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RETRIEVED, golden, 3) == pytest.approx(dcg / idcg)


def test_perfect_and_zero_extremes():
    assert ndcg_at_k(["g1", "g2"], {"g1", "g2"}, 2) == pytest.approx(1.0)
    assert mrr(["g1"], {"g1"}) == 1.0
    assert mrr(["x", "y"], {"g1"}) == 0.0


def test_rank_metrics_notice_position_set_metrics_do_not():
    """The lost-in-the-middle guard: slide the only golden chunk from rank 1
    to rank 5 — hit rate and recall are blind, MRR and nDCG must degrade."""
    golden = {"g"}
    first = ["g", "b", "c", "d", "e"]
    fifth = ["b", "c", "d", "e", "g"]
    assert hit_rate_at_k(first, golden, 5) == hit_rate_at_k(fifth, golden, 5)
    assert recall_at_k(first, golden, 5) == recall_at_k(fifth, golden, 5)
    assert mrr(fifth, golden) < mrr(first, golden)
    assert ndcg_at_k(fifth, golden, 5) < ndcg_at_k(first, golden, 5)


def test_empty_golden_is_undefined_not_zero():
    for fn in (
        lambda: hit_rate_at_k(RETRIEVED, set(), 3),
        lambda: precision_at_k(RETRIEVED, set(), 3),
        lambda: recall_at_k(RETRIEVED, set(), 3),
        lambda: mrr(RETRIEVED, set()),
        lambda: ndcg_at_k(RETRIEVED, set(), 3),
    ):
        with pytest.raises(ValueError):
            fn()
