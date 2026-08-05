"""Dense retrieval: LSA vectors (TF-IDF -> TruncatedSVD), cosine similarity.

Why LSA and not a neural embedding model: this repo values reproducibility
that survives the demo machine — LSA is deterministic, offline, and adds zero
dependencies (scikit-learn is already installed for the experiment sandbox).
The cost is honest and documented: LSA captures co-occurrence semantics, not
deep meaning, so it is a WEAKER dense retriever than a neural model — which
is precisely the condition under which hybrid fusion and reranking earn their
keep, and the eval tables in evals/ measure that claim rather than assert it.

sublinear_tf because section bodies repeat their own key terms; 256 dims (or
n_features-1 on a small corpus) keeps >90% of what matters at this scale.
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

_DIMS = 256


class LsaIndex:
    def __init__(self, texts):
        self._vectorizer = TfidfVectorizer(
            lowercase=True, sublinear_tf=True, stop_words="english",
            token_pattern=r"[A-Za-z0-9_]+",
        )
        tfidf = self._vectorizer.fit_transform(texts)
        dims = min(_DIMS, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        self._svd = TruncatedSVD(n_components=max(dims, 2), random_state=42)
        self._matrix = normalize(self._svd.fit_transform(tfidf))

    def rank(self, query, n):
        """Indices of the top-n documents by cosine similarity, best first."""
        q = self._svd.transform(self._vectorizer.transform([query]))
        q = normalize(q)
        sims = (self._matrix @ q.T).ravel()
        order = np.argsort(-sims, kind="stable")
        return [int(i) for i in order[:n] if sims[i] > 0]
