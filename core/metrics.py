"""The S6E7 competition metric, implemented once and shared by every slice.

Balanced Accuracy = the unweighted mean of per-class recall. Higher is better.
S6E7's target is heavily imbalanced (`at-risk` is ~86% of train), so plain
accuracy would reward predicting the majority class everywhere; balanced
accuracy scores each class equally regardless of its size, which is why
Kaggle picked it and why we must optimise CV against this exact formula and
nothing else.

Written by hand against the definition rather than imported from sklearn: the
project is stdlib-only, and a metric small enough to hand-compute is a metric
we can unit-test against a worked example.
"""


def balanced_accuracy(y_true, y_pred):
    """Unweighted mean of per-class recall over the classes present in y_true.

    Classes that appear only in ``y_pred`` contribute nothing: recall is
    defined over actual members of a class, so a class with no true members
    has no recall to average. This matches sklearn's behaviour.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            "length mismatch: %d true vs %d predicted" % (len(y_true), len(y_pred))
        )
    if not y_true:
        raise ValueError("balanced_accuracy of empty sequences is undefined")

    class_total = {}
    class_hits = {}
    for truth, pred in zip(y_true, y_pred):
        class_total[truth] = class_total.get(truth, 0) + 1
        if truth == pred:
            class_hits[truth] = class_hits.get(truth, 0) + 1

    recalls = [class_hits.get(cls, 0) / total for cls, total in class_total.items()]
    return sum(recalls) / len(recalls)
