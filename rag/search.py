"""The retrieval layer: hybrid BM25 + LSA with RRF, optional LLM rerank.

`search()` is the single seam everything goes through — the agent tool, the
eval harness, and the rerank on/off comparison all call it with different
flags, so the thing the harness measures is the thing the agent runs.

Order contract: the list returned here is RETRIEVER ORDER, best first. The
rank-aware metrics (MRR, nDCG) read this order directly; any
lost-in-the-middle repacking for prompt assembly must happen downstream of
this function, never inside it.
"""

from pathlib import Path

from rag.bm25 import BM25
from rag.chunking import chunk_markdown
from rag.dense import LsaIndex
from rag.fuse import rrf
from rag.rerank import POOL, rerank

REPO_ROOT = Path(__file__).resolve().parents[1]

# The corpus: every committed markdown document that describes the project.
# Ordered and explicit (not a glob) so chunk ids only move when the LIST or a
# document changes, never because the filesystem enumerated differently.
CORPUS_FILES = [
    "README.md",
    "CLAUDE.md",
    "docs/PLAN.md",
    "docs/architecture.md",
    "docs/criterias.md",
    "docs/hw2/shamil.md",
    "docs/milestone1/shamil.md",
    "docs/milestone2/plan.md",
    "docs/milestone2/nikita.md",
    "docs/milestone2/shamil.md",
    "agents/critic_brief.md",
]


class CorpusIndex:
    def __init__(self, root=REPO_ROOT, files=None):
        self.chunks = []
        for rel in files or CORPUS_FILES:
            path = root / rel
            self.chunks.extend(chunk_markdown(rel, path.read_text()))
        texts = [c["text"] for c in self.chunks]
        self._bm25 = BM25(texts)
        self._lsa = LsaIndex(texts)
        self._by_id = {c["id"]: c for c in self.chunks}

    def get(self, chunk_id):
        return self._by_id.get(chunk_id)

    def search(self, query, k=5, use_rerank=False, pool=POOL, stages=None):
        """Top-k chunks for `query`, best first.

        `stages`, if a dict, receives the intermediate rankings (bm25, dense,
        fused ids) — Part 1's before/after analysis reads them; normal
        callers ignore it.
        """
        bm25_idx = self._bm25.rank(query, pool)
        dense_idx = self._lsa.rank(query, pool)
        fused_ids = rrf(
            [[self.chunks[i]["id"] for i in bm25_idx],
             [self.chunks[i]["id"] for i in dense_idx]]
        )[:pool]
        if stages is not None:
            stages["bm25"] = [self.chunks[i]["id"] for i in bm25_idx]
            stages["dense"] = [self.chunks[i]["id"] for i in dense_idx]
            stages["fused"] = list(fused_ids)
        candidates = [self._by_id[cid] for cid in fused_ids]
        if use_rerank:
            candidates = rerank(query, candidates)
            if stages is not None:
                stages["reranked"] = [c["id"] for c in candidates]
        return candidates[:k]


_INDEX = None


def get_index():
    """Process-wide index. Building takes ~1s over ~150 chunks; callers that
    need isolation (tests over fixture corpora) construct CorpusIndex directly."""
    global _INDEX
    if _INDEX is None:
        _INDEX = CorpusIndex()
    return _INDEX
