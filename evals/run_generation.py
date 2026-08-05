"""Run the judged generation metrics over the eval set.

    uv run python -m evals.run_generation             # rerank off
    uv run python -m evals.run_generation --rerank    # both modes

Every (answer, judge) response is cached by rag/llmcall, so the first pass
costs LLM calls and every re-run is free and reproduces exactly. All 24 cases
run in BOTH modes (the assignment allows a subset for rerank-off; the cache
makes the full set affordable, and a full set is the better comparison).
Out-of-corpus cases are judged too — faithfulness and relevance of an honest
refusal is precisely what they exist to test — but context_recall is skipped
for them (no reference answer grounded in the corpus exists).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from evals import generation_metrics as gm
from evals.golden import load_cases
from rag.search import get_index

K = 5
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def evaluate(index, cases, use_rerank):
    per_case = {}
    for case in cases:
        chunks = index.search(case["query"], k=K, use_rerank=use_rerank)
        answer, context = gm.generate_answer(case["query"], chunks)
        scores = {
            "faithfulness": gm.faithfulness(case["query"], answer, context),
            "answer_relevance": gm.answer_relevance(case["query"], answer),
        }
        if case["category"] != "out_of_corpus":
            scores["context_recall"] = gm.context_recall(case["golden_answer"], context)
        per_case[case["id"]] = {
            "category": case["category"],
            "answer": answer,
            "scores": scores,
        }
        print(
            "  %s %s  " % (case["id"], case["category"].ljust(14)),
            " ".join("%s=%.2f" % (m, s["score"]) for m, s in scores.items()),
        )
    return per_case


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def tables(all_per_case):
    metrics = ["faithfulness", "answer_relevance", "context_recall"]
    lines = ["| metric | " + " | ".join(all_per_case) + " |",
             "|---|" + "---|" * len(all_per_case)]
    for m in metrics:
        row = []
        for mode in all_per_case:
            vals = [
                r["scores"][m]["score"]
                for r in all_per_case[mode].values()
                if m in r["scores"]
            ]
            row.append("%.3f (n=%d)" % (_mean(vals), len(vals)))
        lines.append("| %s | " % m + " | ".join(row) + " |")

    cat_lines = ["", "| category | mode | faithfulness | relevance | ctx recall |",
                 "|---|---|---|---|---|"]
    for mode, per_case in all_per_case.items():
        by_cat = defaultdict(list)
        for r in per_case.values():
            by_cat[r["category"]].append(r["scores"])
        for cat in sorted(by_cat):
            rows = by_cat[cat]
            def col(m):
                vals = [s[m]["score"] for s in rows if m in s]
                return "%.3f" % _mean(vals) if vals else "—"
            cat_lines.append("| %s | %s | %s | %s | %s |"
                             % (cat, mode, col("faithfulness"),
                                col("answer_relevance"), col("context_recall")))
    return "\n".join(lines), "\n".join(cat_lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Judged generation metrics.")
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args(argv)

    index = get_index()
    cases = load_cases()
    modes = [("rerank_off", False)] + ([("rerank_on", True)] if args.rerank else [])

    all_per_case = {}
    for label, flag in modes:
        print("== %s" % label)
        all_per_case[label] = evaluate(index, cases, use_rerank=flag)

    overall, by_cat = tables(all_per_case)
    RESULTS_DIR.mkdir(exist_ok=True)
    md = ["# Generation metrics (LLM-as-judge, k=%d)" % K, "",
          "## Overall", "", overall, "", "## By category", by_cat]
    (RESULTS_DIR / "generation.md").write_text("\n".join(md) + "\n")
    (RESULTS_DIR / "generation.json").write_text(json.dumps(all_per_case, indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    main()
