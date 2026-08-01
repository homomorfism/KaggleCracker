"""EDA feeder: inspect the data in the sandbox, let the agent interpret it.

    EDA_SCRIPT (fixed, generic)  ──► sandbox ──► stdout statistics
                                                      │
                                          AgentRuntime.generate
                                                      │
                                          context/eda_findings.md
                                                      │
                                      PromptBuilder.context_docs ──► every node

The split matters. The script computes *statistics* — dtypes, null counts,
cardinality, target relationships, id substructure. It contains no knowledge of
this competition and would run unchanged on any tabular problem. Deciding what
those statistics mean, and what a solution should do about them, is the agent's
job.

That line is the whole point of this feeder. Writing "fill HomePlanet before
converting to category" into the prompt myself would raise the score and prove
nothing: it would demonstrate that I can preprocess the data, which was never in
question. Having the agent derive it from statistics it gathered is a claim about
the system.

The script runs in the same `--network none` sandbox as every node, so EDA gets
the same isolation as training code and needs no separate machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kagglecracker.config import Settings
from kagglecracker.executors.base import Executor, RunStatus
from kagglecracker.runtime.base import AgentRuntime, GenerateResult

EDA_FILENAME = "eda_findings.md"

# Deliberately generic. Every branch keys off dtypes and column names discovered
# at runtime; nothing here knows what competition it is looking at.
EDA_SCRIPT = '''
import pandas as pd, numpy as np, json, itertools

pd.set_option("display.width", 200)
DATA = "/data"

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
sample = pd.read_csv(f"{DATA}/sample_submission.csv")

print("## Shapes")
print(f"train {train.shape} | test {test.shape} | sample_submission {sample.shape}")

# The target is whatever train has that test does not.
target = [c for c in train.columns if c not in test.columns]
target = target[0] if len(target) == 1 else None
print(f"target column: {target}")
print(f"submission columns: {list(sample.columns)}")

print()
print("## Columns")
print(f"{'column':<20} {'dtype':<12} {'nulls':>7} {'null%':>7} {'nunique':>9}  examples")
for col in train.columns:
    s = train[col]
    nulls = int(s.isna().sum())
    ex = [repr(v) for v in s.dropna().unique()[:3]]
    print(f"{col:<20} {str(s.dtype):<12} {nulls:>7} {nulls/len(s):>6.1%} "
          f"{s.nunique():>9}  {', '.join(ex)}")

print()
print("## Null co-occurrence (columns null in the same rows)")
nullable = [c for c in train.columns if train[c].isna().any()]
if len(nullable) > 1:
    nm = train[nullable].isna()
    print(f"rows with at least one null: {nm.any(axis=1).sum()} / {len(train)} "
          f"({nm.any(axis=1).mean():.1%})")
    print(f"rows with all values present: {(~nm.any(axis=1)).sum()}")
else:
    print("fewer than two nullable columns")

if target is not None:
    print()
    print(f"## Target: {target}")
    print(f"dtype {train[target].dtype}")
    vc = train[target].value_counts(normalize=True)
    print(f"distribution: {vc.head(10).round(4).to_dict()}")

    print()
    print("## Relationship to target")
    for col in train.columns:
        if col == target:
            continue
        s = train[col]
        try:
            if s.nunique() <= 12:
                g = train.groupby(col, dropna=False)[target].agg(["mean", "size"])
                print(f"-- {col} (low cardinality)")
                print(g.round(4).to_string())
            elif pd.api.types.is_numeric_dtype(s):
                y = pd.to_numeric(train[target], errors="coerce")
                print(f"-- {col}: corr={s.corr(y):.4f} "
                      f"min={s.min()} med={s.median()} max={s.max()}")
        except Exception as exc:
            print(f"-- {col}: could not summarise ({type(exc).__name__}: {exc})")

print()
print("## High-cardinality / identifier-like columns")
for col in train.columns:
    s = train[col]
    if s.nunique() < len(s) * 0.5 or pd.api.types.is_numeric_dtype(s):
        continue
    print(f"-- {col}: {s.nunique()} unique of {len(s)} rows")
    vals = s.dropna().astype(str)
    # Composite identifiers often hide structure behind a separator. Report it
    # mechanically; whether it matters is a judgement call, not a computation.
    for sep in ["_", "/", "-", " "]:
        if vals.str.contains(sep, regex=False).mean() > 0.9:
            parts = vals.str.split(sep, expand=True)
            print(f"   splits on {sep!r} into {parts.shape[1]} parts")
            for i in range(parts.shape[1]):
                p = parts[i].dropna()
                print(f"     part {i}: {p.nunique()} unique, e.g. {list(p.unique()[:3])}")
                if col in test.columns:
                    tp = test[col].dropna().astype(str).str.split(sep, expand=True)[i]
                    shared = len(set(p) & set(tp.dropna()))
                    print(f"       shared with test: {shared} of {p.nunique()} "
                          f"train values, {tp.nunique()} test values")
            break

print()
print("## Duplicates")
feat = [c for c in train.columns if c != target]
print(f"fully duplicated rows: {train.duplicated().sum()}")
print(f"duplicated on features only: {train.duplicated(subset=feat).sum()}")

print()
print("## Constant / near-constant columns")
flagged = False
for col in train.columns:
    top = train[col].value_counts(normalize=True, dropna=False)
    if len(top) and top.iloc[0] > 0.98:
        print(f"-- {col}: {top.index[0]!r} in {top.iloc[0]:.1%} of rows")
        flagged = True
if not flagged:
    print("none above 98% single-value")

print()
print("## Train vs test drift")
print(f"{'column':<20} {'train':>14} {'test':>14}  note")
for col in test.columns:
    if col not in train.columns:
        continue
    a, b = train[col], test[col]
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        note = ""
        if a.std() and abs(a.mean() - b.mean()) > 0.25 * a.std():
            note = "MEAN SHIFT > 0.25 sd"
        print(f"{col:<20} {a.mean():>14.4f} {b.mean():>14.4f}  {note}")
    else:
        sa, sb = set(a.dropna().astype(str)), set(b.dropna().astype(str))
        only_test = len(sb - sa)
        note = f"{only_test} value(s) only in test" if only_test else ""
        print(f"{col:<20} {len(sa):>14} {len(sb):>14}  {note}")

'''


SUMMARY_PROMPT = """\
Below is the raw output of an exploratory data analysis script run against a
Kaggle tabular competition's training and test data. You did not see the data
itself — only these statistics.

Write `eda_findings.md`: notes for an engineer who is about to write a solution
and has not seen this output. Cover, at minimum:

  - **Columns**: which are features (present in train AND test) and which exist
    only in train — name the target explicitly and say what it is.
  - **Missing values**: which columns, how much, and whether nulls co-occur.
  - **Types and cardinality**: what is numeric, what is categorical, which are
    high-cardinality or identifier-like, which are constant or near-constant.
  - **Signal**: which columns relate to the target and how strongly.
  - **Anything else load-bearing**: train/test drift, duplicates, structure
    hidden inside composite identifiers.

Report only what the statistics below support. Do not speculate about the
domain, and do not invent numbers. If something looks important but the output
is ambiguous, say so rather than guessing.

Return markdown only, no preamble.

---

{eda_output}
"""


@dataclass(frozen=True, slots=True)
class EdaResult:
    ok: bool
    findings_path: Path | None
    raw_output: str
    generation: GenerateResult | None
    error: str = ""


def run_eda(
    *,
    settings: Settings,
    executor: Executor,
    runtime: AgentRuntime,
) -> EdaResult:
    result = executor.run(EDA_SCRIPT, timeout_s=settings.node_timeout_s, node_id="eda")
    if result.status is not RunStatus.OK:
        return EdaResult(
            ok=False,
            findings_path=None,
            raw_output=result.stdout_tail,
            generation=None,
            error=f"EDA script failed ({result.status}): {result.error_text[:500]}",
        )

    raw = (result.stdout_path.read_text() if result.stdout_path else result.stdout_tail)

    gen = runtime.generate(SUMMARY_PROMPT.format(eda_output=raw), action="draft")

    # The summariser returns prose, so `code` is empty and the normal
    # OK/NO_CODE classification does not apply. Only a truncated or refused
    # response is a real failure here.
    text = gen.raw_text.strip()
    if not text:
        return EdaResult(
            ok=False,
            findings_path=None,
            raw_output=raw,
            generation=gen,
            error=f"summariser returned nothing ({gen.status}): {gen.error_text[:300]}",
        )

    settings.context_dir.mkdir(parents=True, exist_ok=True)
    path = settings.context_dir / EDA_FILENAME
    path.write_text(text + "\n")
    return EdaResult(ok=True, findings_path=path, raw_output=raw, generation=gen)
