"""Entry point the server spawns: run the demo analysis for one project.

Demo = the REAL recon tools running on the project's REAL uploaded data,
driven by a scripted FakeModel instead of a live LLM. Deterministic and free:
every number the UI charts is genuinely computed from the user's files; only
the prose is canned. The live-model driver will replace exactly one argument
(model) later — journal, tools and loop stay identical.

Run from the repo root:  python -m ui.run_analysis <slug> <run_id> [--pace 0.8]
"""

import argparse
import os
import time

from core.registry import Registry
from tests.fakemodel import FakeModel, Reply, ToolCall
from tools.recon import plans, profile
from tools.recon.paths import plan_path
from ui import projects
from ui.journal import JournalWriter
from ui.live_model import DEFAULT_MODEL, AnthropicModel
from ui.runner import run_with_journal

_SAMPLE_ROWS = 50_000

# How long a live run waits at the gate for the human in the UI. Generous,
# because a person may be reading the preview; but bounded, because a run
# must not hang forever — and the timeout answer is "", a denial.
_GATE_TIMEOUT = 600.0


def _paced(model, seconds):
    """Sleep before each reply so a demo run unfolds at watchable speed
    instead of finishing before the first UI poll."""

    def call(messages, tools):
        if seconds:
            time.sleep(seconds)
        return model(messages, tools)

    return call


def _pick_files(names):
    """(train, compare): prefer conventional names, fall back to first files."""
    train = next((n for n in names if "train" in n.lower()), names[0])
    others = [n for n in names if n != train]
    compare = next((n for n in others if "test" in n.lower()), others[0] if others else None)
    return train, compare


def _build_replies(train, compare, target, plan_exists):
    replies = [
        Reply(
            text=(
                "Starting with structure: missingness, column types and "
                "cardinality of %s tell us how much cleaning the plan needs." % train
            ),
            tool_calls=[
                ToolCall(
                    "profile_dataset",
                    {
                        "path": train,
                        "checks": ["missingness", "dtypes", "cardinality"],
                        "sample_rows": _SAMPLE_ROWS,
                    },
                )
            ],
        )
    ]

    if target:
        replies.append(
            Reply(
                text=(
                    "Now the modelling view of %s: class balance of %r, numeric "
                    "ranges, and which columns move with the target." % (train, target)
                ),
                tool_calls=[
                    ToolCall(
                        "profile_dataset",
                        {
                            "path": train,
                            "checks": [
                                "target_balance",
                                "numeric_summary",
                                "correlation_with_target",
                            ],
                            "target": target,
                            "sample_rows": _SAMPLE_ROWS,
                        },
                    )
                ],
            )
        )
    else:
        replies.append(
            Reply(
                text=(
                    "No target column is set for this project, so the modelling "
                    "view is numeric ranges only."
                ),
                tool_calls=[
                    ToolCall(
                        "profile_dataset",
                        {
                            "path": train,
                            "checks": ["numeric_summary"],
                            "sample_rows": _SAMPLE_ROWS,
                        },
                    )
                ],
            )
        )

    if compare:
        replies.append(
            Reply(
                text=(
                    "Comparing %s against %s for drift: a feature whose "
                    "distribution shifts between the files can wreck validation."
                    % (train, compare)
                ),
                tool_calls=[
                    ToolCall(
                        "profile_dataset",
                        {
                            "path": train,
                            "checks": ["train_test_drift"],
                            "compare_path": compare,
                            "sample_rows": _SAMPLE_ROWS,
                        },
                    )
                ],
            )
        )

    # Deliberate probe for an optional file, identical whether or not it was
    # uploaded. Present: it profiles for real. Absent: the not_found error
    # branch shows up in the run timeline — exactly what the UI should render.
    replies.append(
        Reply(
            text=(
                "Checking whether an original-source dataset (original.csv) was "
                "uploaded alongside the competition files."
            ),
            tool_calls=[
                ToolCall(
                    "profile_dataset",
                    {
                        "path": "original.csv",
                        "checks": ["missingness"],
                        "sample_rows": _SAMPLE_ROWS,
                    },
                )
            ],
        )
    )

    plan_tool = "overwrite_preprocessing_plan" if plan_exists else "write_preprocessing_plan"
    replies.append(
        Reply(
            text=(
                "Replacing the existing preprocessing plan with one grounded in "
                "this run's profile — this needs human approval."
                if plan_exists
                else "Writing the first preprocessing plan grounded in this run's profile."
            ),
            tool_calls=[
                ToolCall(
                    plan_tool,
                    {"dataset": train, "plan_markdown": _plan_markdown(train, target)},
                )
            ],
        )
    )

    replies.append(Reply(text=_final_report(train, target)))
    return replies


def _plan_markdown(train, target):
    lines = [
        "# Preprocessing plan — %s" % train,
        "",
        "Grounded in this run's profile report (the run journal holds the numbers).",
        "",
        "1. **Missingness** — impute numeric columns with the median, categorical",
        "   with an explicit 'missing' category; drop nothing until a model asks.",
        "2. **Mixed-type columns** — any column the dtypes check flagged as mixed",
        "   is parsed once, failures becoming missing values, before encoding.",
        "3. **Identifiers** — columns the cardinality check marked likely_id are",
        "   excluded from features.",
        "4. **Imbalance** — if target_balance shows heavy skew, evaluate with",
        "   balanced accuracy and try class weights before any resampling.",
        "5. **Drift** — columns flagged by train_test_drift get grouped or",
        "   temporal validation splits rather than a plain shuffle.",
    ]
    if target:
        # Slot it before the blank line that already follows the intro.
        lines.insert(3, "Target column: `%s`." % target)
    return "\n".join(lines)


def _final_report(train, target):
    view = (
        "the modelling view (class balance and correlations for target %r)" % target
        if target
        else "the modelling view (numeric ranges; no target was set)"
    )
    return "\n".join(
        [
            "# Data reconnaissance report — %s" % train,
            "",
            "This run profiled the uploaded data directly: structure (missingness,",
            "types, cardinality), %s," % view,
            "and file-to-file drift where a comparison file was available.",
            "",
            "Every number in the run's charts comes from those tool results; the",
            "findings store inside the project workspace holds the same rows for",
            "later runs to query. The preprocessing plan derived from this profile",
            "is saved under plans/.",
        ]
    )


def run_demo_analysis(slug, run_id, pace=0.0):
    project = projects.get_project(slug)
    project_path = projects.project_dir(slug)
    rdir = projects.run_dir(slug, run_id)
    rdir.mkdir(parents=True, exist_ok=True)
    writer = JournalWriter(rdir / "journal.jsonl")

    names = [f["name"] for f in project["files"]]
    if not names:
        writer.write(
            "run_failed",
            reason="no csv files uploaded; add data before starting an analysis",
        )
        writer.close()
        raise ValueError("project %r has no data files" % slug)

    # The recon tools resolve every path through KC_WORKSPACE, so pointing it
    # at the project directory is the whole multi-project mechanism. Restored
    # afterwards because in-process callers (tests) share this environment.
    previous = os.environ.get("KC_WORKSPACE")
    os.environ["KC_WORKSPACE"] = str(project_path)
    try:
        train, compare = _pick_files(names)
        replies = _build_replies(
            train,
            compare,
            project.get("target") or "",
            plan_exists=plan_path(train).exists(),
        )
        registry = Registry()
        profile.register(registry)
        plans.register(registry)

        prompt = "TASK: data reconnaissance for project %r." % project["name"]
        if project.get("description"):
            prompt += " " + project["description"]

        final = run_with_journal(
            [prompt],
            _paced(FakeModel(replies), pace),
            registry,
            writer,
            # Demo runs approve their own gate so the full flow is visible
            # unattended; the journal marks the answer scripted=True so a
            # rendered transcript never passes it off as a human decision.
            gate_answer_fn=lambda: "yes",
        )
        (rdir / "report.md").write_text(final)
        return final
    finally:
        writer.close()
        if previous is None:
            os.environ.pop("KC_WORKSPACE", None)
        else:
            os.environ["KC_WORKSPACE"] = previous


def wait_for_gate_answer(run_path, timeout=_GATE_TIMEOUT, poll=1.0):
    """Build the answer_fn a live run hands to the gate: block until the UI
    writes gate_answer.txt next to the journal, consume it, return its text.

    Timing out returns "" — a denial by the gate's own rules. Absence of a
    human is never consent, so a walked-away human cannot approve anything.
    """

    def answer_fn():
        target = run_path / "gate_answer.txt"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if target.is_file():
                answer = target.read_text().strip()
                target.unlink()  # consumed: a second gate needs its own answer
                return answer
            time.sleep(poll)
        return ""

    return answer_fn


def _live_task(project, names):
    target = project.get("target") or ""
    lines = [
        "You are the data-reconnaissance agent for a Kaggle-style project.",
        "Project: %s." % project["name"],
    ]
    if project.get("description"):
        lines.append("Task description from the human: %s" % project["description"])
    lines.append("CSV files available under workspace/data: %s." % ", ".join(names))
    if target:
        lines.append("Target column: %r." % target)
    else:
        lines.append("No target column is set; skip target-dependent checks.")
    lines += [
        "",
        "Do this, in order:",
        "1. Profile the training file with profile_dataset: structure first",
        "   (missingness, dtypes, cardinality), then the modelling view",
        "   (target_balance, numeric_summary, correlation_with_target when a",
        "   target exists), then train_test_drift against the test file when",
        "   one exists. Never repeat a check you already ran on the same file.",
        "2. Write a preprocessing plan grounded ONLY in what the profile",
        "   reports actually said (write_preprocessing_plan, or",
        "   overwrite_preprocessing_plan if one already exists).",
        "3. Then STOP: reply with no tool calls, and make that final message a",
        "   markdown data report (# heading, short paragraphs) describing what",
        "   the data looks like and what a modeller must watch out for.",
        "A tool error is information, not a dead end: adjust the call or move on.",
    ]
    return "\n".join(lines)


def run_live_analysis(slug, run_id, model_id=None):
    """The same journal/loop/tools as the demo, with two swaps: the model is
    live, and the gate waits for a real human answer from the UI."""
    project = projects.get_project(slug)
    project_path = projects.project_dir(slug)
    rdir = projects.run_dir(slug, run_id)
    rdir.mkdir(parents=True, exist_ok=True)
    writer = JournalWriter(rdir / "journal.jsonl")

    names = [f["name"] for f in project["files"]]
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    fail = None
    if not names:
        fail = "no csv files uploaded; add data before starting an analysis"
    elif not api_key:
        fail = "ANTHROPIC_API_KEY is not set in the server environment"
    if fail:
        writer.write("run_failed", reason=fail)
        writer.close()
        raise ValueError(fail)

    previous = os.environ.get("KC_WORKSPACE")
    os.environ["KC_WORKSPACE"] = str(project_path)
    try:
        task = _live_task(project, names)
        registry = Registry()
        profile.register(registry)
        plans.register(registry)
        final = run_with_journal(
            # messages starts empty: the live adapter keeps the task in its
            # native API conversation, so prompt= carries it to the journal.
            [],
            AnthropicModel(api_key, task, model=model_id or DEFAULT_MODEL),
            registry,
            writer,
            max_steps=16,
            gate_answer_fn=wait_for_gate_answer(rdir),
            gate_scripted=False,
            prompt=task,
        )
        if final:
            (rdir / "report.md").write_text(final)
        return final
    finally:
        writer.close()
        if previous is None:
            os.environ.pop("KC_WORKSPACE", None)
        else:
            os.environ["KC_WORKSPACE"] = previous


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run one analysis for a UI project.")
    parser.add_argument("slug")
    parser.add_argument("run_id")
    parser.add_argument("--mode", choices=("demo", "live"), default="demo")
    parser.add_argument("--pace", type=float, default=0.8, help="demo mode only")
    parser.add_argument("--model", default=None, help="live mode only")
    args = parser.parse_args(argv)
    if args.mode == "live":
        run_live_analysis(args.slug, args.run_id, model_id=args.model)
    else:
        run_demo_analysis(args.slug, args.run_id, pace=args.pace)


if __name__ == "__main__":
    main()
