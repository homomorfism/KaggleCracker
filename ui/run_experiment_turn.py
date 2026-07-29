"""One experiment-agent chat turn.

    python -m ui.run_experiment_turn <slug> <exp_id>

Same stateless-turn design as ui/run_eda.py: the conversation file is the
durable thing, each turn is a fresh subprocess primed from it. What is
different is the brief — this agent plans BEFORE it codes. Its workflow is
question -> plan -> human approval -> implement, and the phase transitions are
carried by tools (propose_experiment_plan, finalize_experiment) so the
experiment record moves only when the model commits to something concrete.

The lock directory holds this process's pid, so the API's stop endpoint can
kill a runaway turn; the conversation survives and the human resumes later.
"""

import argparse
import json
import os
import shutil
import sys

from core.registry import Registry
from tools.exec import experiments as exec_tools
from tools.exec import shell
from tools.experiments import planning
from ui import experiments, projects
from ui.journal import JournalWriter, read_events
from ui.live_model import DEFAULT_MODEL, AnthropicModel
from ui.runner import run_with_journal

_HISTORY_WINDOW = 20


def _eda_insights(project_dir):
    """The dashboard's titles, commentary and stats — the distilled EDA."""
    path = project_dir / "eda" / "dashboard.json"
    if not path.is_file():
        return "(no EDA dashboard yet)"
    try:
        panels = json.loads(path.read_text()).get("panels", [])
    except ValueError:
        return "(no EDA dashboard yet)"
    lines = []
    for p in panels:
        line = "- %s" % p.get("title", p.get("id", "?"))
        if p.get("type") == "stat":
            line += ": %s" % p.get("data", {}).get("value", "")
        if p.get("commentary"):
            line += " — %s" % p["commentary"]
        lines.append(line)
    return "\n".join(lines) or "(dashboard is empty)"


def _community_knowledge(project_dir):
    """The observer's briefing plus per-notebook technique lists."""
    parts = []
    summary = project_dir / "knowledge" / "summary.md"
    if summary.is_file():
        parts.append(summary.read_text()[:4000])
    store = project_dir / "knowledge" / "notebooks.jsonl"
    if store.is_file():
        techniques = []
        for line in store.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            names = ", ".join(t.get("name", "?") for t in r.get("techniques", []))
            claim = r.get("cv_claim") or r.get("lb_claim") or ""
            techniques.append(
                "- %s (%s votes%s): %s"
                % (r.get("title", "?"), r.get("votes", 0), " · " + claim if claim else "", names)
            )
        if techniques:
            parts.append("Notebook techniques:\n" + "\n".join(techniques[:15]))
    return "\n\n".join(parts) or "(no community knowledge collected yet)"


def _task(slug, exp_id, project, meta, messages):
    project_dir = projects.project_dir(slug)
    try:
        comp = projects.competition_meta(slug)
    except FileNotFoundError:
        comp = {}
    plan_path = experiments.experiment_dir(slug, exp_id) / "plan.md"
    plan = plan_path.read_text() if plan_path.is_file() else ""
    code_dir = experiments.experiment_dir(slug, exp_id) / "code"
    attempts = sorted(p.name for p in code_dir.glob("*.py")) if code_dir.is_dir() else []
    files = ", ".join(f["name"] for f in project["files"]) or "(none)"

    lines = [
        "You are the experiment agent for a Kaggle competition project. You design",
        "ML experiments WITH the human, then implement them in the sandbox.",
        "",
        "Competition: %s" % (comp.get("title") or project["name"]),
        "Evaluation metric: %s" % (comp.get("evaluation_metric") or "(unknown)"),
        "Target column: %r" % (project.get("target") or "(not set — ask)"),
        "Files under data/: %s" % files,
        "",
        "== Dataset insights (from the EDA dashboard) ==",
        _eda_insights(project_dir),
        "",
        "== Community knowledge (from the notebook observer) ==",
        _community_knowledge(project_dir),
        "",
        "== This experiment ==",
        "id: %s | state: %s | name: %s" % (exp_id, meta["state"], meta["name"]),
    ]
    if plan:
        lines += ["Current plan:", plan[:3000]]
    if attempts:
        lines.append("Code attempts already run: %s" % ", ".join(attempts))
    lines += [
        "",
        "== Conversation (oldest first) ==",
    ]
    for m in messages[-_HISTORY_WINDOW:]:
        lines.append("%s: %s" % (m["role"].upper(), m["text"]))
    lines += [
        "",
        "== Workflow — follow the phase the state puts you in ==",
        "1. DRAFT: if real decisions are open (imputation strategy, model family,",
        "   validation scheme, feature choices), ask the human — ONE message, at",
        "   most 4 crisp numbered questions, no tool calls, then stop. If the",
        "   insights above already answer a question, do not ask it.",
        "   run_experiment's dataset_ref must be one of the EXACT file names",
        "   listed under data/ above (e.g. 'train.csv') — never a made-up name.",
        "2. When the open questions are settled, call propose_experiment_plan",
        "   with a short name and a markdown plan (sections: Data preprocessing,",
        "   Model, Validation, Expected outcome). Then stop with one sentence",
        "   asking for approval.",
        "3. PLAN_PROPOSED: if the human's latest message approves (yes / accept /",
        "   go / looks good), implement: write ONE self-contained Python script",
        "   per attempt for run_experiment (pandas, numpy, scikit-learn,",
        "   lightgbm, xgboost are installed; anything else: run_bash with",
        "   'pip install <pkg>' — network is available; read data with relative",
        "   paths like 'data/train.csv'; the script MUST print",
        "   'CV_SCORE: <float>'). A tool error is information —",
        "   fix the script and retry. When a score is in: call",
        "   record_experiment_result, then finalize_experiment, then stop with a",
        "   2-3 sentence result summary. If the human wants changes instead,",
        "   revise with propose_experiment_plan.",
        "Never run experiments before the plan is approved. Never invent scores.",
    ]
    return "\n".join(lines)


def run_turn(slug, exp_id, model_id=None, model_factory=None, mode="auto"):
    project = projects.get_project(slug)
    project_dir = projects.project_dir(slug)
    exp_dir = experiments.experiment_dir(slug, exp_id)
    meta = experiments.get_meta(slug, exp_id)
    messages = experiments.read_messages(slug, exp_id)
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("no pending user message for experiment %r" % exp_id)

    turn_dir = experiments.new_turn_dir(slug, exp_id)
    writer = JournalWriter(turn_dir / "journal.jsonl")

    task = _task(slug, exp_id, project, meta, messages)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if model_factory is not None:
        model = model_factory(task)
    elif mode == "demo" or (mode == "auto" and not api_key):
        # No key needed: scripted prose, but the training and the score are
        # real — the demo model reads the sandbox result out of the transcript.
        from ui.demo_experiment import DemoExperimentModel

        csvs = [f["name"] for f in project["files"] if f["name"].endswith(".csv")]
        train = next((n for n in csvs if "train" in n.lower()), csvs[0] if csvs else "")
        model = DemoExperimentModel(
            meta, messages[-1]["text"], project.get("target") or "", train
        )
    else:
        if not api_key:
            writer.write("run_failed", reason="ANTHROPIC_API_KEY is not set")
            writer.close()
            raise ValueError("ANTHROPIC_API_KEY is not set")
        # 16k tokens: training scripts ride inside tool_use blocks and the
        # recon default of 3000 truncates them (see ui/run_eda.py).
        model = AnthropicModel(api_key, task, model=model_id or DEFAULT_MODEL, max_tokens=16000)

    previous_ws = os.environ.get("KC_WORKSPACE")
    previous_exp = os.environ.get("KC_EXPERIMENT_DIR")
    os.environ["KC_WORKSPACE"] = str(project_dir)
    os.environ["KC_EXPERIMENT_DIR"] = str(exp_dir)
    try:
        registry = Registry()
        exec_tools.register(registry)
        planning.register(registry)
        shell.register(registry)
        final = run_with_journal(
            [],
            model,
            registry,
            writer,
            # Implementation turns burn steps on script retries; planning turns
            # stop after one reply anyway.
            max_steps=24,
            prompt=messages[-1]["text"],
            # Debugging a training script means consecutive tool errors are
            # normal; the recon default of 2 would disable run_experiment mid-fix.
            max_consecutive_fails=6,
        )
        experiments.append_message(slug, exp_id, "assistant", final or "(done)")
        return final
    except Exception as e:
        experiments.append_message(
            slug, exp_id, "assistant", "The turn failed: %s" % str(e)[:300]
        )
        raise
    finally:
        writer.close()
        # The journal is complete now — mirror the executed scripts into code/.
        experiments.save_code_attempts(
            slug, exp_id, read_events(turn_dir / "journal.jsonl")
        )
        for var, prev in (("KC_WORKSPACE", previous_ws), ("KC_EXPERIMENT_DIR", previous_exp)):
            if prev is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = prev


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run one experiment-agent turn.")
    parser.add_argument("slug")
    parser.add_argument("exp_id")
    parser.add_argument("--model", default=None)
    # auto: live when ANTHROPIC_API_KEY is set, scripted demo otherwise.
    parser.add_argument("--mode", choices=("auto", "demo", "live"), default="auto")
    args = parser.parse_args(argv)

    exp_dir = experiments.experiment_dir(args.slug, args.exp_id)
    lock = exp_dir / "lock"
    try:
        lock.mkdir()
    except FileExistsError:
        print("a turn is already running for %s" % args.exp_id)
        return 1
    # The pid is the stop endpoint's handle on this process.
    (lock / "pid").write_text(str(os.getpid()))
    # Lock claimed: the spawn-pending marker has done its job.
    (exp_dir / "spawn_pending").unlink(missing_ok=True)
    try:
        run_turn(args.slug, args.exp_id, model_id=args.model, mode=args.mode)
        return 0
    except (ValueError, RuntimeError) as e:
        print("experiment turn failed: %s" % e)
        return 1
    finally:
        shutil.rmtree(lock, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
