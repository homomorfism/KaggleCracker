# Retrieval metrics

Corpus: 123 chunks · 24 eval cases (22 scored, 2 out-of-corpus excluded: c20, c21)

## Overall (mean over scored cases)

| metric | rerank_off |
|---|---|
| hit_rate@3 | 0.727 |
| hit_rate@5 | 0.818 |
| hit_rate@10 | 0.955 |
| precision@3 | 0.273 |
| precision@5 | 0.236 |
| precision@10 | 0.155 |
| recall@3 | 0.572 |
| recall@5 | 0.719 |
| recall@10 | 0.802 |
| ndcg@3 | 0.502 |
| ndcg@5 | 0.567 |
| ndcg@10 | 0.602 |
| mrr | 0.560 |

## By category — rerank_off

| category | n | hit_rate@5 | recall@5 | mrr | ndcg@5 |
|---|---|---|---|---|---|
| acronym | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_term | 5 | 1.000 | 0.900 | 0.667 | 0.695 |
| multi_hop | 2 | 0.500 | 0.417 | 0.556 | 0.500 |
| near_duplicate | 2 | 0.500 | 0.500 | 0.250 | 0.325 |
| numeric_fact | 5 | 0.800 | 0.629 | 0.396 | 0.426 |
| paraphrase | 6 | 0.833 | 0.722 | 0.565 | 0.535 |

## Stage analysis (which stage first hit a golden chunk, top-5)

| case | category | dense@5 | bm25@5 | fused@5 | reranked@5 |
|---|---|---|---|---|---|
| c01 | exact_term | ✓ | ✓ | ✓ | · |
| c02 | exact_term | ✓ | ✓ | ✓ | · |
| c03 | exact_term | ✓ | ✓ | ✓ | · |
| c04 | acronym | ✓ | ✓ | ✓ | · |
| c05 | acronym | ✓ | ✓ | ✓ | · |
| c06 | numeric_fact | — | — | — | · |
| c07 | numeric_fact | ✓ | ✓ | ✓ | · |
| c08 | numeric_fact | ✓ | ✓ | ✓ | · |
| c09 | numeric_fact | — | — | ✓ | · |
| c10 | paraphrase | ✓ | ✓ | ✓ | · |
| c11 | paraphrase | — | ✓ | — | · |
| c12 | paraphrase | — | ✓ | ✓ | · |
| c13 | paraphrase | ✓ | ✓ | ✓ | · |
| c14 | multi_hop | — | — | — | · |
| c15 | near_duplicate | — | — | — | · |
| c16 | near_duplicate | ✓ | ✓ | ✓ | · |
| c17 | exact_term | ✓ | ✓ | ✓ | · |
| c18 | multi_hop | ✓ | ✓ | ✓ | · |
| c19 | paraphrase | ✓ | ✓ | ✓ | · |
| c22 | exact_term | ✓ | ✓ | ✓ | · |
| c23 | numeric_fact | ✓ | — | ✓ | · |
| c24 | paraphrase | ✓ | ✓ | ✓ | · |
