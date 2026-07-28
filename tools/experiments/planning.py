"""The experiment agent's lifecycle tools: propose a plan, close the loop.

The exec slice's run_experiment / record_experiment_result do the actual
training work; these two tools move the experiment's own record through its
states (draft -> plan_proposed -> finished). Which experiment they act on
comes from KC_EXPERIMENT_DIR — set by the turn runner the same way
KC_WORKSPACE is, so the tools stay argument-free about identity and the model
cannot write into a different experiment's record.

Both tools are reversible (a plan can be re-proposed, a finish overwritten by
a later finalize), so neither is gated.
"""

import json
import os
import time
from pathlib import Path

from core.contracts import err, ok
from core.registry import ToolSpec


def _experiment_dir():
    d = os.environ.get("KC_EXPERIMENT_DIR")
    if not d:
        raise RuntimeError("KC_EXPERIMENT_DIR is not set")
    return Path(d)


def _load_meta(d):
    return json.loads((d / "experiment.json").read_text())


def _save_meta(d, meta):
    meta["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp = d / "experiment.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, d / "experiment.json")


def propose_experiment_plan(args):
    name = args["name"].strip()
    plan = args["plan_markdown"].strip()
    if not name:
        return err("bad_input", "name is empty")
    if len(plan) < 100:
        return err(
            "bad_input",
            "plan_markdown is too thin (%d chars); a reviewable plan covers data "
            "preprocessing, model, validation, and expected metric" % len(plan),
            retryable=True,
        )
    d = _experiment_dir()
    (d / "plan.md").write_text(plan)
    meta = _load_meta(d)
    meta["name"] = name
    meta["state"] = "plan_proposed"
    _save_meta(d, meta)
    return ok(state="plan_proposed", name=name)


def finalize_experiment(args):
    d = _experiment_dir()
    meta = _load_meta(d)
    if not (d / "plan.md").is_file():
        return err(
            "bad_input",
            "no plan was ever proposed; call propose_experiment_plan and get the "
            "human's approval before finalizing",
        )
    meta["state"] = "finished"
    meta["cv_score"] = args["cv_score"]
    meta["result_summary"] = args["summary"].strip()
    _save_meta(d, meta)
    return ok(state="finished", cv_score=args["cv_score"])


PROPOSE_PLAN = ToolSpec(
    name="propose_experiment_plan",
    description=(
        "Save this experiment's plan for human review and set the experiment to "
        "plan_proposed. Call it once your open questions are answered and you "
        "have a concrete plan: preprocessing steps, model choice, validation "
        "scheme, expected metric. Calling it again replaces the plan (do that "
        "after feedback). Do NOT call it while questions to the human are still "
        "unanswered, and do NOT call it to record results — it never runs code."
    ),
    parameters={
        "name": {"type": "str", "required": True},
        "plan_markdown": {"type": "str", "required": True},
    },
    fn=propose_experiment_plan,
)

FINALIZE = ToolSpec(
    name="finalize_experiment",
    description=(
        "Mark this experiment finished with its final CV score and a short "
        "result summary. Call it exactly once, after run_experiment produced the "
        "score and record_experiment_result filed it on the leaderboard. Do NOT "
        "call it for failed or abandoned attempts — leave those in their "
        "current state and tell the human instead."
    ),
    parameters={
        "cv_score": {"type": "float", "required": True},
        "summary": {"type": "str", "required": True},
    },
    fn=finalize_experiment,
)


def register(registry):
    """Register the experiment-lifecycle tools into the given registry."""
    registry.register(PROPOSE_PLAN)
    registry.register(FINALIZE)
    return registry
