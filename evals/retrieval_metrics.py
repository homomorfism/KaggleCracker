"""The five rank-aware retrieval metrics, hand-rolled on purpose.

No evaluation library ships this family (DeepEval and Ragas lack all five;
their ContextualPrecision/Recall are judged metrics measuring something
else), so these are ~60 lines of plain Python with unit tests against
hand-computed values.

Contract: `retrieved` is RETRIEVER ORDER, best first — MRR and nDCG read
position, and feeding them a lost-in-the-middle repacked order silently
degrades exactly those two while the set-based three look fine. `golden` is
a set of relevant chunk ids. Cases with an EMPTY golden set (deliberately
unanswerable) are undefined here, not zero: callers must exclude them from
averages and report their count separately.
"""

import math


def _require_golden(golden):
    if not golden:
        raise ValueError(
            "metric undefined for an empty golden set — exclude the case, do not score it"
        )


def hit_rate_at_k(retrieved, golden, k):
    """1.0 if ANY relevant chunk made the top-k cut, else 0.0."""
    _require_golden(golden)
    return 1.0 if any(r in golden for r in retrieved[:k]) else 0.0


def precision_at_k(retrieved, golden, k):
    """Fraction of the top-k that is relevant. Divides by k, not by how many
    were returned: returning fewer than k is the retriever's problem."""
    _require_golden(golden)
    return sum(1 for r in retrieved[:k] if r in golden) / k


def recall_at_k(retrieved, golden, k):
    """Fraction of the relevant chunks that made the top-k cut."""
    _require_golden(golden)
    return sum(1 for r in retrieved[:k] if r in golden) / len(golden)


def mrr(retrieved, golden):
    """1/rank of the FIRST relevant chunk; 0.0 if none was retrieved at all."""
    _require_golden(golden)
    for i, r in enumerate(retrieved):
        if r in golden:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved, golden, k):
    """Binary-relevance nDCG@k with the standard log2 discount — the metric
    that notices a relevant chunk sliding from position 1 to position 5."""
    _require_golden(golden)
    dcg = sum(
        1.0 / math.log2(i + 2) for i, r in enumerate(retrieved[:k]) if r in golden
    )
    ideal_hits = min(len(golden), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


ALL = {
    "hit_rate": lambda r, g, k: hit_rate_at_k(r, g, k),
    "precision": lambda r, g, k: precision_at_k(r, g, k),
    "recall": lambda r, g, k: recall_at_k(r, g, k),
    "mrr": lambda r, g, k: mrr(r, g),
    "ndcg": lambda r, g, k: ndcg_at_k(r, g, k),
}
