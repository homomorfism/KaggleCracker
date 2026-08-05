"""Reciprocal Rank Fusion.

RRF_score(d) = sum over rankings of 1 / (K + rank_of_d). K=60 is the value
from the original Cormack et al. paper and the one the class used; on this
corpus the tables were insensitive to K in [20, 100] (a ~150-chunk corpus
rarely produces deep rank disagreements), so the conventional constant stays
— documented rather than tuned into noise.
"""

RRF_K = 60


def rrf(rankings, k=RRF_K):
    """Fuse ranked lists of ids into one list, best first.

    Ties break by (first appearance list order, then id) so fusion is fully
    deterministic — metric runs must reproduce bit-for-bit.
    """
    scores = {}
    first_seen = {}
    for li, ranking in enumerate(rankings):
        for rank, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
            first_seen.setdefault(doc, (li, rank))
    return sorted(scores, key=lambda d: (-scores[d], first_seen[d], str(d)))
