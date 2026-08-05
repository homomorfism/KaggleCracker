"""Run the rank-aware retrieval metrics over the eval set.

    uv run python -m evals.run_retrieval            # rerank off (free, offline)
    uv run python -m evals.run_retrieval --rerank   # both off and on

Outputs evals/results/retrieval.md (tables for the report) and retrieval.json
(raw numbers). Empty-golden cases are excluded from every average and counted
separately — a metric over "deliberately unanswerable" is undefined, not zero.
The stage table is Part 1's before/after evidence: for each case, which
pipeline stage first got a golden chunk into its top-5.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from evals import retrieval_metrics as rm
from evals.golden import load_cases, resolve_golden
from rag.search import get_index

KS = (3, 5, 10)
STAGE_K = 5
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def evaluate(index, cases, use_rerank):
    """Per-case rankings + metric values. The ranking is retrieved once at
    depth max(KS) and every k reads a prefix of that same order."""
    per_case = {}
    skipped = []
    for case in cases:
        golden = resolve_golden(case, index)
        if not golden:
            skipped.append(case["id"])
            continue
        stages = {}
        retrieved = [
            c["id"]
            for c in index.search(
                case["query"], k=max(KS), use_rerank=use_rerank, stages=stages
            )
        ]
        metrics = {"mrr": rm.mrr(retrieved, golden)}
        for k in KS:
            metrics["hit_rate@%d" % k] = rm.hit_rate_at_k(retrieved, golden, k)
            metrics["precision@%d" % k] = rm.precision_at_k(retrieved, golden, k)
            metrics["recall@%d" % k] = rm.recall_at_k(retrieved, golden, k)
            metrics["ndcg@%d" % k] = rm.ndcg_at_k(retrieved, golden, k)
        per_case[case["id"]] = {
            "category": case["category"],
            "golden": sorted(golden),
            "retrieved": retrieved,
            "stages": stages,
            "metrics": metrics,
        }
    return per_case, skipped


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def aggregate(per_case):
    overall = defaultdict(list)
    by_category = defaultdict(lambda: defaultdict(list))
    for record in per_case.values():
        for name, value in record["metrics"].items():
            overall[name].append(value)
            by_category[record["category"]][name].append(value)
    return (
        {name: _mean(vals) for name, vals in overall.items()},
        {
            cat: {name: _mean(vals) for name, vals in metrics.items()}
            for cat, metrics in by_category.items()
        },
    )


def _hit(ids, golden, k):
    return "✓" if any(i in golden for i in ids[:k]) else "—"


def stage_table(per_case):
    """Which stage first captured a golden chunk in its top-5 — the
    qualitative before/after, with the cases nothing fixed left visible."""
    lines = [
        "| case | category | dense@5 | bm25@5 | fused@5 | reranked@5 |",
        "|---|---|---|---|---|---|",
    ]
    for cid in sorted(per_case):
        r = per_case[cid]
        golden = set(r["golden"])
        s = r["stages"]
        lines.append(
            "| %s | %s | %s | %s | %s | %s |"
            % (
                cid,
                r["category"],
                _hit(s.get("dense", []), golden, STAGE_K),
                _hit(s.get("bm25", []), golden, STAGE_K),
                _hit(s.get("fused", []), golden, STAGE_K),
                _hit(s.get("reranked", []), golden, STAGE_K)
                if "reranked" in s
                else "·",
            )
        )
    return "\n".join(lines)


def metrics_table(results):
    """results: {mode_label: overall_metrics}"""
    names = ["hit_rate", "precision", "recall", "ndcg"]
    lines = ["| metric | " + " | ".join(results) + " |",
             "|---|" + "---|" * len(results)]
    for name in names:
        for k in KS:
            key = "%s@%d" % (name, k)
            lines.append(
                "| %s | " % key
                + " | ".join("%.3f" % results[m][key] for m in results)
                + " |"
            )
    lines.append(
        "| mrr | " + " | ".join("%.3f" % results[m]["mrr"] for m in results) + " |"
    )
    return "\n".join(lines)


def category_table(per_case, key_metrics=("hit_rate@5", "recall@5", "mrr", "ndcg@5")):
    by_cat = defaultdict(list)
    for record in per_case.values():
        by_cat[record["category"]].append(record["metrics"])
    lines = ["| category | n | " + " | ".join(key_metrics) + " |",
             "|---|---|" + "---|" * len(key_metrics)]
    for cat in sorted(by_cat):
        rows = by_cat[cat]
        lines.append(
            "| %s | %d | " % (cat, len(rows))
            + " | ".join("%.3f" % _mean([r[k] for r in rows]) for k in key_metrics)
            + " |"
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rank-aware retrieval metrics.")
    parser.add_argument("--rerank", action="store_true",
                        help="also run the rerank-on pass (LLM calls, cached)")
    args = parser.parse_args(argv)

    index = get_index()
    cases = load_cases()
    modes = [("rerank_off", False)] + ([("rerank_on", True)] if args.rerank else [])

    all_results, all_per_case, skipped = {}, {}, []
    for label, flag in modes:
        per_case, skipped = evaluate(index, cases, use_rerank=flag)
        overall, by_cat = aggregate(per_case)
        all_results[label] = overall
        all_per_case[label] = per_case

    RESULTS_DIR.mkdir(exist_ok=True)
    counts = defaultdict(int)
    for case in cases:
        counts[case["category"]] += 1

    md = ["# Retrieval metrics", ""]
    md.append("Corpus: %d chunks · %d eval cases (%d scored, %d out-of-corpus excluded: %s)"
              % (len(index.chunks), len(cases), len(cases) - len(skipped),
                 len(skipped), ", ".join(skipped)))
    md += ["", "## Overall (mean over scored cases)", "", metrics_table(all_results)]
    for label in all_results:
        md += ["", "## By category — %s" % label, "",
               category_table(all_per_case[label])]
    base = list(all_per_case.values())[-1]
    md += ["", "## Stage analysis (which stage first hit a golden chunk, top-%d)" % STAGE_K,
           "", stage_table(base)]
    (RESULTS_DIR / "retrieval.md").write_text("\n".join(md) + "\n")
    (RESULTS_DIR / "retrieval.json").write_text(json.dumps(
        {"results": all_results,
         "per_case": all_per_case,
         "skipped": skipped}, indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    main()
