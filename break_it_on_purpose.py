"""Regenerate the two comparison transcripts that make requirement 4 visible.

The same scenario runs twice against the tests/fixtures/no_score.py fixture — a
script that exits cleanly but never prints the required 'CV_SCORE: <float>' line,
printing a 'mean auc = 0.8814' line instead:

  * WITH the error branch: the real run_experiment returns a bad_input envelope
    (no CV_SCORE line), the loop takes its error branch, and the agent refuses to
    record anything.
  * WITHOUT the error branch: a naive run_experiment hands stdout back as data
    and reports the last float it finds (0.8814) as the CV score, so the agent
    confidently records a number the experiment never produced.

The model is the SAME reactive function in both runs: it records the score when
the run succeeded and reports failure otherwise. Only what the tool told it
differs — that is the whole point. Transcripts land in runs/.

Run from the repo root:  python break_it_on_purpose.py
"""

import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from core.contracts import ok
from core.errors import StepLimitReached
from core.loop import run_agent
from core.registry import Registry, ToolSpec
from tests.fakemodel import Reply, ToolCall
from tools.exec.experiments import RECORD_RESULT, RUN_EXPERIMENT, _run_precheck

ROOT = Path(__file__).resolve().parent
NO_SCORE_CODE = (ROOT / "tests" / "fixtures" / "no_score.py").read_text()
RUN_ARGS = {"code": NO_SCORE_CODE, "dataset_ref": "train.csv", "timeout_s": 30, "cv_folds": 5}


def _mute(*a, **k):
    return None


def reactive_model(messages, tools):
    """The 'model'. It observes the last tool result and reacts to it. This exact
    logic runs in both scenarios; the only thing that changes between runs is what
    the run_experiment tool reported back."""
    last = messages[-1] if messages else ""

    # Observe: nothing has run yet -> act by running the experiment.
    if not (isinstance(last, str) and last.startswith("[TOOL ")):
        return Reply(tool_calls=[ToolCall("run_experiment", dict(RUN_ARGS))])

    # Reason + act, branching purely on what the tool told us:
    if last.startswith("[TOOL OK run_experiment]"):
        # The run reported success, so trust its score and record it.
        data = ast.literal_eval(last.split("] ", 1)[1])
        return Reply(tool_calls=[ToolCall("record_experiment_result", {
            "experiment_id": data["experiment_id"],
            "cv_score": data["cv_score"],
            "fold_scores": [data["cv_score"]],
        })])
    if last.startswith("[TOOL ERROR run_experiment]"):
        # The run reported failure as its own branch -> do NOT record anything.
        return Reply(text="run_experiment reported failure; NOT recording a score.")
    if last.startswith("[TOOL OK record_experiment_result]"):
        return Reply(text="Recorded the CV score and stopped.")

    # Any other surface -> stop rather than guess.
    return Reply(text="Unexpected tool result; stopping: %s" % last)


def naive_run_experiment(args):
    """The DANGEROUS alternative, with no error branch. It runs the script, takes
    whatever landed on stdout as data, and reports the last float it can find as
    the CV score — even when the script never printed one. A broken run becomes a
    confident, wrong number instead of an error the loop could branch on."""
    ws = Path(os.environ.get("KC_WORKSPACE", "workspace"))
    exp_dir = ws / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    exp_id = uuid.uuid4().hex[:12]
    script = exp_dir / (exp_id + ".py")
    script.write_text(args["code"])
    proc = subprocess.run(
        [sys.executable, str(script.relative_to(ws))],
        cwd=str(ws), capture_output=True, text=True,
    )
    floats = re.findall(r"-?\d+\.\d+", proc.stdout)
    score = float(floats[-1]) if floats else 0.0  # last float, no questions asked
    return ok(experiment_id=exp_id, cv_score=score, cv_folds=args["cv_folds"],
              stdout_tail=proc.stdout.strip()[-800:])


# Same name, same parameters, same precheck as the real tool — only the body
# (and its missing error branch) differs, so the two runs are a fair comparison.
NAIVE_RUN = ToolSpec(
    name="run_experiment",
    description=RUN_EXPERIMENT.description,
    parameters=RUN_EXPERIMENT.parameters,
    fn=naive_run_experiment,
    precheck=_run_precheck,
)


def run_scenario(run_spec):
    """Run the reactive model to completion against one run_experiment spec in a
    throwaway workspace. Returns (messages, final_text)."""
    ws = Path(tempfile.mkdtemp(prefix="kc_break_"))
    try:
        (ws / "data").mkdir(parents=True)
        (ws / "data" / "train.csv").write_text("id,label\n1,0\n2,1\n")
        os.environ["KC_WORKSPACE"] = str(ws)

        reg = Registry()
        reg.register(run_spec)
        reg.register(RECORD_RESULT)

        messages = []
        try:
            final = run_agent(messages, reactive_model, reg, max_steps=8,
                              input_fn=_mute, output_fn=_mute)
        except StepLimitReached as e:
            messages = e.messages
            final = "<step limit reached>"
        return messages, final
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def _short(args):
    """Abbreviate long string args (the experiment code) so the transcript stays
    readable."""
    return {k: (v[:37] + "..." if isinstance(v, str) and len(v) > 40 else v)
            for k, v in args.items()}


def render(title, note, messages, final):
    lines = [title, "=" * len(title), note, ""]
    for m in messages:
        if isinstance(m, str):
            lines.append(m)  # a tool result line, already formatted by tool_message
        elif m.tool_calls:
            calls = ", ".join("%s(%s)" % (c.name, _short(c.args)) for c in m.tool_calls)
            lines.append("[MODEL] act  -> %s" % calls)
        else:
            lines.append("[MODEL] stop -> %s" % m.text)
    lines += ["", "[FINAL] %s" % final]
    return "\n".join(lines) + "\n"


def main():
    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)

    msgs_with, final_with = run_scenario(RUN_EXPERIMENT)
    (runs / "with_branch.txt").write_text(render(
        "WITH the error branch (real run_experiment)",
        "The tool returns bad_input when no CV_SCORE line is printed, so the loop "
        "takes its error branch and the agent refuses to record a score.",
        msgs_with, final_with,
    ))

    msgs_without, final_without = run_scenario(NAIVE_RUN)
    (runs / "without_branch.txt").write_text(render(
        "WITHOUT the error branch (naive run_experiment)",
        "The naive tool returns stdout as data and reports the last float it finds "
        "(0.8814, from a 'mean auc' line) as the CV score, so the agent records a "
        "number the experiment never actually produced.",
        msgs_without, final_without,
    ))

    print("wrote runs/with_branch.txt and runs/without_branch.txt")
    print("  with branch    -> %s" % final_with)
    print("  without branch -> %s" % final_without)


if __name__ == "__main__":
    main()
