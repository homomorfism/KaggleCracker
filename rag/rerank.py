"""LLM reranker over the fused candidate pool.

Depth: the pipeline retrieves POOL=20 fused candidates and reranks all of
them in ONE model call (the chunks ride in a single numbered prompt), then
the caller cuts to top-k. What the extra latency buys: RRF only knows each
retriever's rank, not *why* a chunk ranked — the reranker reads the actual
text against the actual question, which is where near-duplicate noise and
"right terms, wrong section" candidates get demoted. One call per query
(~2-4s cold, 0s cached) versus per-chunk calls at 20x the cost.

Scores are parsed strictly; a malformed judge reply raises rather than
silently falling back to the fused order — a rerank that quietly didn't
happen would poison the rerank-on metric tables.
"""

import json
import re

from rag.llmcall import complete

POOL = 20

_SYSTEM = (
    "You are a retrieval reranker. Score how useful each numbered passage is "
    "for answering the query: 0 = irrelevant, 10 = directly answers it. "
    "Judge only usefulness for THIS query; ignore style and length. Reply "
    "with ONLY a JSON object mapping passage numbers to integer scores, like "
    '{"1": 7, "2": 0}.'
)


def rerank(query, chunks):
    """Return `chunks` reordered best-first. Ties keep the fused order, so
    the reranker can only ever refine the fusion, not scramble it."""
    if len(chunks) <= 1:
        return list(chunks)
    numbered = "\n\n".join(
        "[%d] %s" % (i + 1, c["text"][:800]) for i, c in enumerate(chunks)
    )
    reply = complete(_SYSTEM, "Query: %s\n\nPassages:\n\n%s" % (query, numbered))
    match = re.search(r"\{[^{}]*\}", reply, re.DOTALL)
    if not match:
        raise ValueError("reranker returned no JSON object: %r" % reply[:200])
    raw = json.loads(match.group(0))
    scores = {}
    for key, value in raw.items():
        scores[int(key)] = float(value)
    order = sorted(
        range(len(chunks)), key=lambda i: (-scores.get(i + 1, 0.0), i)
    )
    return [chunks[i] for i in order]
