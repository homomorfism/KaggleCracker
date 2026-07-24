"""Regenerate the two comparison transcripts for the recon slice's requirement 4.

The same scenario runs twice against tests/fixtures/ragged.csv — a CSV whose
line 5 has two fields where the header promises three:

  * WITH the error branch: the real profile_dataset refuses the non-rectangle
    with a bad_input naming the exact line, the loop takes its error branch,
    and the agent declines to write a plan for a file it could not read.
  * WITHOUT the error branch: a naive profiler force-fits every row to the
    header (pad with '', truncate the excess) and reports confident statistics
    about data the file never contained — so the agent writes a confident,
    wrong preprocessing plan, and every downstream experiment inherits it.

The model is the SAME reactive function in both runs: it writes a plan when
profiling succeeded and reports failure otherwise. Only what the tool told it
differs — that is the whole point. Transcripts land in runs/.

Run from the repo root:  python break_it_on_purpose_recon.py
"""

import ast
import csv
import os
import shutil
import tempfile
from pathlib import Path

from core.contracts import ok
from core.errors import StepLimitReached
from core.loop import run_agent
from core.registry import Registry, ToolSpec
from tests.fakemodel import Reply, ToolCall
from tools.recon.plans import WRITE_PLAN
from tools.recon.profile import (
    PROFILE_DATASET,
    _check_dtypes,
    _check_missingness,
    _columns,
    _profile_precheck,
)

ROOT = Path(__file__).resolve().parent
RAGGED = ROOT / "tests" / "fixtures" / "ragged.csv"
PROFILE_ARGS = {"path": "ragged.csv", "checks": ["missingness", "dtypes"]}


def _mute(*a, **k):
    return None


def _plan_from(report):
    """Turn a profile report into plan lines — mechanically, so the transcript
    shows exactly which 'facts' the plan was built on."""
    lines = ["# preprocessing plan for ragged.csv", ""]
    for name, m in report.get("missingness", {}).items():
        if m["missing"]:
            lines.append("- impute %r: %d missing (%.0f%%)" % (name, m["missing"], m["pct"] * 100))
    for name, d in report.get("dtypes", {}).items():
        if d.get("mixed"):
            lines.append("- coerce %r to numeric; %d text values are junk" % (name, d["text"]))
    lines.append("- rows profiled: %d" % report.get("rows_profiled", 0))
    return "\n".join(lines) + "\n"


def reactive_model(messages, tools):
    """The 'model'. It observes the last tool result and reacts to it. This
    exact logic runs in both scenarios; the only thing that changes between
    runs is what profile_dataset reported back."""
    last = messages[-1] if messages else ""

    # Observe: nothing has run yet -> act by profiling the dataset.
    if not (isinstance(last, str) and last.startswith("[TOOL ")):
        return Reply(tool_calls=[ToolCall("profile_dataset", dict(PROFILE_ARGS))])

    # Reason + act, branching purely on what the tool told us:
    if last.startswith("[TOOL OK profile_dataset]"):
        # Profiling reported success, so trust its numbers and plan from them.
        report = ast.literal_eval(last.split("] ", 1)[1])
        return Reply(tool_calls=[ToolCall("write_preprocessing_plan", {
            "dataset": "ragged.csv",
            "plan_markdown": _plan_from(report),
        })])
    if last.startswith("[TOOL ERROR profile_dataset]"):
        # Profiling reported failure as its own branch -> do NOT plan from it.
        return Reply(text="profile_dataset reported failure; NOT writing a plan "
                          "for a file I could not actually read.")
    if last.startswith("[TOOL OK write_preprocessing_plan]"):
        return Reply(text="Preprocessing plan written; recon done.")

    # Any other surface -> stop rather than guess.
    return Reply(text="Unexpected tool result; stopping: %s" % last)


def naive_profile_dataset(args):
    """The DANGEROUS alternative, with no structural error branch. Every row is
    force-fitted to the header — short rows padded with '', long rows cut — and
    the statistics are then computed by the REAL check functions. The numbers
    look just as authoritative; they describe a file that does not exist."""
    ws = Path(os.environ.get("KC_WORKSPACE", "workspace"))
    with (ws / "data" / args["path"]).open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [(row + [""] * len(header))[:len(header)] for row in reader]
    cols = _columns(header, rows)
    report = {"path": args["path"], "rows_profiled": len(rows), "columns": header}
    if "missingness" in args["checks"]:
        report["missingness"] = _check_missingness(cols)
    if "dtypes" in args["checks"]:
        report["dtypes"] = _check_dtypes(cols)
    return ok(**report)


# Same name, same parameters, same precheck as the real tool — only the body
# (and its missing error branch) differs, so the two runs are a fair comparison.
NAIVE_PROFILE = ToolSpec(
    name="profile_dataset",
    description=PROFILE_DATASET.description,
    parameters=PROFILE_DATASET.parameters,
    fn=naive_profile_dataset,
    precheck=_profile_precheck,
)


def run_scenario(profile_spec):
    """Run the reactive model to completion against one profile_dataset spec in
    a throwaway workspace. Returns (messages, final_text, plan_text_or_None)."""
    ws = Path(tempfile.mkdtemp(prefix="kc_break_recon_"))
    try:
        (ws / "data").mkdir(parents=True)
        shutil.copy(RAGGED, ws / "data" / "ragged.csv")
        os.environ["KC_WORKSPACE"] = str(ws)

        reg = Registry()
        reg.register(profile_spec)
        reg.register(WRITE_PLAN)

        messages = []
        try:
            final = run_agent(messages, reactive_model, reg, max_steps=8,
                              input_fn=_mute, output_fn=_mute)
        except StepLimitReached as e:
            messages = e.messages
            final = "<step limit reached>"

        plan = ws / "plans" / "ragged.csv.plan.md"
        plan_text = plan.read_text() if plan.exists() else None
        # The throwaway workspace's absolute tmp path is noise in a committed
        # transcript; render it as the stable name so regenerations diff clean.
        messages = [m.replace(str(ws), "workspace") if isinstance(m, str) else m
                    for m in messages]
        return messages, final, plan_text
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def render(title, note, messages, final, plan_text):
    lines = [title, "=" * len(title), note, ""]
    for m in messages:
        if isinstance(m, str):
            lines.append(m)  # a tool result line, already formatted by tool_message
        elif m.tool_calls:
            calls = ", ".join("%s(%s)" % (c.name, c.args) for c in m.tool_calls)
            lines.append("[MODEL] act  -> %s" % calls)
        else:
            lines.append("[MODEL] stop -> %s" % m.text)
    lines += ["", "[FINAL] %s" % final]
    if plan_text is None:
        lines.append("[PLAN FILE] none written")
    else:
        lines.append("[PLAN FILE] workspace/plans/ragged.csv.plan.md:")
        lines += ["  " + l for l in plan_text.splitlines()]
    return "\n".join(lines) + "\n"


def main():
    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)

    msgs_with, final_with, plan_with = run_scenario(PROFILE_DATASET)
    (runs / "recon_with_branch.txt").write_text(render(
        "WITH the error branch (real profile_dataset)",
        "The tool refuses the ragged file with bad_input naming line 5, so the "
        "loop takes its error branch and the agent writes no plan.",
        msgs_with, final_with, plan_with,
    ))

    msgs_without, final_without, plan_without = run_scenario(NAIVE_PROFILE)
    (runs / "recon_without_branch.txt").write_text(render(
        "WITHOUT the error branch (naive profile_dataset)",
        "The naive tool force-fits every row to the header and reports confident "
        "statistics about rows the file never contained, so the agent writes a "
        "confident, wrong preprocessing plan from a file it never actually read.",
        msgs_without, final_without, plan_without,
    ))

    print("wrote runs/recon_with_branch.txt and runs/recon_without_branch.txt")
    print("  with branch    -> %s" % final_with)
    print("  without branch -> %s" % final_without)


if __name__ == "__main__":
    main()
