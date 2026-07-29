"""Execute the preprocessing plan on the project's data — step 2's demo run.

Deterministic, no model: the decisions come straight from the findings the
recon run recorded, which is the whole point — the plan is grounded in the
profile, so a mechanical executor can follow it. Every decision is journaled
so the prep page can show the work; prepared CSVs land in <project>/prep/
next to a manifest.json describing what was done.

Run from the repo root:  python -m ui.run_prep <slug>
"""

import argparse
import json
import os
import shutil
import time

import pandas as pd

from ui import projects
from ui.journal import JournalWriter

# The same tokens the profiler treats as missing, so the executor and the
# findings can never disagree about what "missing" means.
_NA_TOKENS = ["", "na", "NA", "n/a", "N/A", "null", "NULL", "none", "None", "nan", "NaN"]


def _decisions(findings, target):
    """Flagged findings -> concrete column actions. Drop wins over the rest;
    the target column is never touched."""
    drop, impute, coerce, notes = [], [], [], []
    for f in findings:
        col = f["column_name"]
        value = f["value"]
        check = f["check_name"]
        if col == "*" or col == target:
            continue
        if check == "cardinality" and (value.get("likely_id") or value.get("constant")):
            drop.append((col, "identifier" if value.get("likely_id") else "constant"))
        elif check == "dtypes" and value.get("mixed"):
            coerce.append(col)
        elif check == "missingness" and value.get("missing", 0) > 0:
            impute.append(col)
        elif check == "train_test_drift":
            # Drift is a validation decision, not a file transformation: the
            # plan says grouped/temporal splits, so it is surfaced, not edited.
            notes.append(col)
    dropped = {c for c, _ in drop}
    impute = [c for c in impute if c not in dropped]
    coerce = [c for c in coerce if c not in dropped]
    notes = [c for c in notes if c not in dropped]
    return drop, impute, coerce, notes


def _fill_value(series):
    """Median for numeric columns, mode for the rest — matching the plan."""
    if pd.api.types.is_numeric_dtype(series):
        return float(series.median())
    mode = series.mode(dropna=True)
    return str(mode.iloc[0]) if len(mode) else "missing"


def _prepare_frame(df, drop, impute, coerce, fills):
    for col, _why in drop:
        if col in df.columns:
            df = df.drop(columns=[col])
    for col in coerce:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col, value in fills.items():
        if col in df.columns:
            df[col] = df[col].fillna(value)
    return df


def _pick_train(names):
    return next((n for n in names if "train" in n.lower()), names[0])


class _PacedWriter:
    """Journal writer that pauses after each event so a watching UI sees the
    steps land one by one instead of a finished wall. Pace 0 = no theatre."""

    def __init__(self, writer, pace):
        self._writer = writer
        self._pace = pace

    def write(self, event_type, **fields):
        event = self._writer.write(event_type, **fields)
        if self._pace:
            time.sleep(self._pace)
        return event


def run_prep(slug, pace=0.0):
    project = projects.get_project(slug)
    d = projects.project_dir(slug)
    prep = d / "prep"
    prep.mkdir(exist_ok=True)

    lock = prep / "lock"
    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError("a preparation run is already in progress for %r" % slug)
    (prep / "spawn_pending").unlink(missing_ok=True)

    writer = JournalWriter(prep / "journal.jsonl")
    try:
        _run(slug, project, d, prep, _PacedWriter(writer, pace))
    except Exception as e:
        writer.write("run_failed", reason="%s: %s" % (type(e).__name__, e))
        raise
    finally:
        writer.close()
        shutil.rmtree(lock, ignore_errors=True)


def _run(slug, project, d, prep, writer):
    target = project.get("target") or ""
    names = [f["name"] for f in project["files"] if f["name"].endswith(".csv")]
    plans = projects.list_plans(slug)
    findings = projects.list_findings(slug)  # flagged only: the decisions

    writer.write(
        "run_started",
        text="Executing the preprocessing plan on %d csv file%s."
        % (len(names), "" if len(names) == 1 else "s"),
    )
    if not plans:
        raise ValueError("no preprocessing plan yet — run a recon analysis first")
    if not names:
        raise ValueError("no csv files to prepare")

    drop, impute, coerce, drift_notes = _decisions(findings, target)
    for col, why in drop:
        writer.write("step", action="drop", column=col,
                     text="drop %r — %s, useless as a feature" % (col, why))
    for col in coerce:
        writer.write("step", action="coerce", column=col,
                     text="coerce %r to numeric — mixed types; failures become missing" % col)
    for col in impute:
        writer.write("step", action="impute", column=col,
                     text="impute %r — median if numeric, mode otherwise" % col)
    for col in drift_notes:
        writer.write("step", action="note", column=col,
                     text="%r drifts between files — validate with grouped splits, not edited here" % col)
    if not (drop or impute or coerce):
        writer.write("step", action="note",
                     text="no flagged findings require changes — files pass through cleaned copies")

    # Fill values come from the TRAIN file only, then apply everywhere: the
    # test file must never leak its own statistics into preparation.
    train = _pick_train(names)
    train_df = pd.read_csv(d / "data" / train, na_values=_NA_TOKENS, keep_default_na=True)
    for col in coerce:
        if col in train_df.columns:
            train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
    fills = {}
    for col in set(impute) | set(coerce):
        if col in train_df.columns:
            fills[col] = _fill_value(train_df[col])

    outputs = []
    for name in names:
        df = pd.read_csv(d / "data" / name, na_values=_NA_TOKENS, keep_default_na=True)
        df = _prepare_frame(df, drop, impute, coerce, fills)
        out_name = name[: -len(".csv")] + ".prepared.csv"
        out_path = prep / out_name
        df.to_csv(out_path, index=False)
        record = {
            "name": out_name,
            "source": name,
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "bytes": out_path.stat().st_size,
        }
        outputs.append(record)
        writer.write("file_written", **record)

    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train": train,
        "decisions": {
            "dropped": [{"column": c, "why": w} for c, w in drop],
            "coerced": coerce,
            "imputed": [{"column": c, "fill": fills.get(c)} for c in sorted(set(impute) | set(coerce)) if c in fills],
            "drift_notes": drift_notes,
        },
        "files": outputs,
    }
    tmp = prep / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, prep / "manifest.json")

    writer.write(
        "run_finished",
        text="Prepared %d file%s: dropped %d column%s, coerced %d, imputed %d."
        % (
            len(outputs), "" if len(outputs) == 1 else "s",
            len(drop), "" if len(drop) == 1 else "s",
            len(coerce), len(fills),
        ),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Execute the preprocessing plan for a project.")
    parser.add_argument("slug")
    parser.add_argument("--pace", type=float, default=0.6)
    args = parser.parse_args(argv)
    run_prep(args.slug, pace=args.pace)


if __name__ == "__main__":
    main()
