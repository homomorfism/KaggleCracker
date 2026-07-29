"""The demo experiment agent: scripted phases, real sandbox training."""

import json

import pytest

from ui import experiments, projects
from ui.run_experiment_turn import run_turn


def _train_csv():
    # x1 fully determines y, so the real CV score lands high — proof the
    # sandbox actually trained rather than a canned number appearing.
    rows = ["id,x1,x2,y"]
    for i in range(30):
        x1 = i % 10
        rows.append("%d,%d,%d,%s" % (i + 1, x1, (i * 7) % 13, "sick" if x1 < 4 else "fit"))
    return ("\n".join(rows) + "\n").encode()


def _project():
    projects.create_project("Demo", "predict y", "y")
    projects.save_data_file("demo", "train.csv", _train_csv())


def test_demo_experiment_full_cycle(ui_projects_tmp, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _project()
    exp = experiments.create_experiment("demo", "")

    # Turn 1: the demo agent proposes a plan and stops for the human.
    run_turn("demo", exp["id"])
    meta = experiments.get_meta("demo", exp["id"])
    assert meta["state"] == "plan_proposed"
    assert (experiments.experiment_dir("demo", exp["id"]) / "plan.md").is_file()
    assert experiments.read_messages("demo", exp["id"])[-1]["role"] == "assistant"

    # Turn 2: approval -> real training -> record -> finalize.
    experiments.append_message("demo", exp["id"], "user", "yes")
    run_turn("demo", exp["id"])
    meta = experiments.get_meta("demo", exp["id"])
    assert meta["state"] == "finished"
    assert isinstance(meta["cv_score"], float) and 0.0 <= meta["cv_score"] <= 1.0

    # The score on the leaderboard is the sandbox's, with real fold scores.
    lb = projects.project_dir("demo") / "experiments" / "leaderboard.jsonl"
    row = json.loads(lb.read_text().splitlines()[-1])
    assert row["cv_score"] == pytest.approx(meta["cv_score"])
    assert len(row["fold_scores"]) == 3

    # The exact executed script was mirrored into code/ for the human to read.
    code = sorted((experiments.experiment_dir("demo", exp["id"]) / "code").glob("*.py"))
    assert code and "CV_SCORE" in code[0].read_text()


def test_demo_agent_waits_for_explicit_approval(ui_projects_tmp, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _project()
    exp = experiments.create_experiment("demo", "")
    run_turn("demo", exp["id"])

    # A vague reply is not consent: nothing trains, nothing is recorded.
    experiments.append_message("demo", exp["id"], "user", "hmm, not sure")
    run_turn("demo", exp["id"])
    assert experiments.get_meta("demo", exp["id"])["state"] == "plan_proposed"
    assert not (projects.project_dir("demo") / "experiments" / "leaderboard.jsonl").exists()
