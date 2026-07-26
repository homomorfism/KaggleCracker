"""Tests for core/metrics.py.

The hand-computed example is the point of the file: if the metric is wrong,
every CV score in every experiment is wrong, and no other test would notice —
they all just compare our numbers against our numbers.
"""

import pytest

from core.metrics import balanced_accuracy


def test_hand_computed_example():
    # Worked by hand, three classes with S6E7's shape (one dominant class):
    #   at-risk:   6 true, 5 predicted correctly -> recall 5/6
    #   unhealthy: 2 true, 1 predicted correctly -> recall 1/2
    #   fit:       2 true, 2 predicted correctly -> recall 2/2
    # balanced accuracy = (5/6 + 1/2 + 1) / 3 = (14/6) / 3 = 7/9
    y_true = ["at-risk"] * 6 + ["unhealthy"] * 2 + ["fit"] * 2
    y_pred = ["at-risk"] * 5 + ["fit"] + ["unhealthy", "at-risk"] + ["fit", "fit"]
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(7 / 9)


def test_perfect_prediction_scores_one():
    y = ["fit", "unhealthy", "at-risk", "at-risk"]
    assert balanced_accuracy(y, list(y)) == 1.0


def test_majority_class_prediction_scores_one_over_n_classes():
    # The imbalance trap the metric exists to punish: predicting the dominant
    # class everywhere gives recall 1 on it and 0 elsewhere -> 1/3, not 0.86.
    y_true = ["at-risk"] * 86 + ["unhealthy"] * 8 + ["fit"] * 6
    y_pred = ["at-risk"] * 100
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(1 / 3)


def test_class_only_in_predictions_is_ignored():
    # "typo-class" never occurs in y_true, so it has no recall to average.
    y_true = ["fit", "fit", "unhealthy"]
    y_pred = ["fit", "typo-class", "unhealthy"]
    assert balanced_accuracy(y_true, y_pred) == pytest.approx((0.5 + 1.0) / 2)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        balanced_accuracy(["fit"], ["fit", "fit"])


def test_empty_input_raises():
    with pytest.raises(ValueError):
        balanced_accuracy([], [])
