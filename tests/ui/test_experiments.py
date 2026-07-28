"""The experiment agent end to end with FakeModel: question -> plan ->
approval -> real sandboxed run -> finished record, plus the HTTP surface."""

import json

import pytest
from fastapi.testclient import TestClient

import ui.app
import ui.run_experiment_turn as turn_runner
from tests.fakemodel import FakeModel, Reply, ToolCall
from ui import experiments, projects

_CSV = "id,x1,y\n1,0.5,0\n2,0.9,1\n3,0.4,0\n4,1.2,1\n"

_SCRIPT = "print('CV_SCORE: 0.8123')\n"

_PLAN = (
    "# Plan\n\n## Data preprocessing\nmedian imputation\n\n## Model\nlogistic "
    "regression baseline\n\n## Validation\n5-fold stratified CV\n\n## Expected "
    "outcome\naccuracy around 0.8\n"
)


@pytest.fixture
def project():
    p = projects.create_project("titanic", competition="titanic")
    (projects.project_dir("titanic") / "data" / "train.csv").write_text(_CSV)
    return p


def _turn(slug, exp_id, replies):
    model = FakeModel(replies)
    return turn_runner.run_turn(slug, exp_id, model_factory=lambda task: model)


def test_store_lifecycle(project):
    meta = experiments.create_experiment("titanic", "try gradient boosting")
    assert meta["state"] == "draft"
    # The prompt seeds the conversation, so the first turn has work to do.
    msgs = experiments.read_messages("titanic", meta["id"])
    assert msgs[0]["role"] == "user" and "gradient boosting" in msgs[0]["text"]

    # An empty prompt becomes the "you decide" brief, never an empty message.
    meta2 = experiments.create_experiment("titanic", "")
    first = experiments.read_messages("titanic", meta2["id"])[0]
    assert "yourself" in first["text"]

    listed = experiments.list_experiments("titanic")
    assert [e["id"] for e in listed] == sorted([meta["id"], meta2["id"]])
    assert all(e["running"] is False for e in listed)


def test_question_plan_approve_run_finish(project):
    meta = experiments.create_experiment("titanic", "baseline please")
    exp_id = meta["id"]

    # Turn 1: the agent asks instead of guessing — no tools, just a question.
    _turn("titanic", exp_id, [Reply(text="1. Which validation scheme do you prefer?")])
    msgs = experiments.read_messages("titanic", exp_id)
    assert msgs[-1]["role"] == "assistant" and "validation" in msgs[-1]["text"]
    assert experiments.get_meta("titanic", exp_id)["state"] == "draft"

    # Turn 2: answered -> the agent commits to a plan.
    experiments.append_message("titanic", exp_id, "user", "stratified 5-fold")
    _turn(
        "titanic",
        exp_id,
        [
            Reply(
                text="planning",
                tool_calls=[
                    ToolCall(
                        "propose_experiment_plan",
                        {"name": "logreg baseline", "plan_markdown": _PLAN},
                    )
                ],
            ),
            Reply(text="Plan is up — approve?"),
        ],
    )
    meta = experiments.get_meta("titanic", exp_id)
    assert meta["state"] == "plan_proposed" and meta["name"] == "logreg baseline"
    assert "median imputation" in (
        experiments.experiment_dir("titanic", exp_id) / "plan.md"
    ).read_text()

    # Turn 3: approval -> a real subprocess run, filed and finalized.
    experiments.append_message("titanic", exp_id, "user", "accept")
    _turn(
        "titanic",
        exp_id,
        [
            Reply(
                text="running",
                tool_calls=[
                    ToolCall(
                        "run_experiment",
                        {"code": _SCRIPT, "dataset_ref": "train.csv"},
                    )
                ],
            ),
            Reply(
                text="recording",
                tool_calls=[
                    ToolCall(
                        "record_experiment_result",
                        {
                            "experiment_id": "x1",
                            "cv_score": 0.8123,
                            "fold_scores": [0.8, 0.82],
                        },
                    ),
                    ToolCall(
                        "finalize_experiment",
                        {"cv_score": 0.8123, "summary": "baseline lands at 0.81"},
                    ),
                ],
            ),
            Reply(text="Done: CV 0.8123."),
        ],
    )
    meta = experiments.get_meta("titanic", exp_id)
    assert meta["state"] == "finished" and meta["cv_score"] == 0.8123

    # The executed script is readable, extracted from the journal.
    detail = experiments.get_detail("titanic", exp_id)
    assert detail["code"] and detail["code"][0]["source"] == _SCRIPT
    # And the run landed on the shared pipeline leaderboard.
    lb = projects.project_dir("titanic") / "experiments" / "leaderboard.jsonl"
    assert json.loads(lb.read_text().splitlines()[0])["cv_score"] == 0.8123


def test_finalize_requires_a_plan(project):
    meta = experiments.create_experiment("titanic", "hasty")
    _turn(
        "titanic",
        meta["id"],
        [
            Reply(
                text="skipping ahead",
                tool_calls=[ToolCall("finalize_experiment", {"cv_score": 0.9, "summary": "s"})],
            ),
            Reply(text="ok, I need a plan first"),
        ],
    )
    # The premature finalize bounced; the experiment is still a draft.
    assert experiments.get_meta("titanic", meta["id"])["state"] == "draft"


def test_thin_plan_is_rejected(project):
    meta = experiments.create_experiment("titanic", "thin")
    _turn(
        "titanic",
        meta["id"],
        [
            Reply(
                text="planning",
                tool_calls=[
                    ToolCall("propose_experiment_plan", {"name": "x", "plan_markdown": "do stuff"})
                ],
            ),
            Reply(text="fine, I will write a real plan"),
        ],
    )
    assert experiments.get_meta("titanic", meta["id"])["state"] == "draft"


def test_experiment_routes(project, monkeypatch):
    api = TestClient(ui.app.app, raise_server_exceptions=False)
    spawned = []
    monkeypatch.setattr(ui.app, "_spawn", lambda mod, args, log: spawned.append((mod, args)))

    r = api.post("/api/projects/titanic/experiments", json={"prompt": "try xgboost"})
    assert r.status_code == 201
    exp_id = r.json()["id"]
    assert r.json()["running"] is True
    assert spawned == [("ui.run_experiment_turn", ["titanic", exp_id])]

    listing = api.get("/api/projects/titanic/experiments").json()["experiments"]
    assert [e["id"] for e in listing] == [exp_id]

    detail = api.get("/api/projects/titanic/experiments/%s" % exp_id).json()
    assert detail["experiment"]["state"] == "draft"
    assert detail["messages"][0]["text"] == "try xgboost"
    assert detail["plan_md"] == "" and detail["code"] == []

    # Chat: 409 while locked, spawn when free.
    lock = experiments.experiment_dir("titanic", exp_id) / "lock"
    lock.mkdir()
    r = api.post(
        "/api/projects/titanic/experiments/%s/messages" % exp_id, json={"text": "hi"}
    )
    assert r.status_code == 409
    lock.rmdir()
    r = api.post(
        "/api/projects/titanic/experiments/%s/messages" % exp_id, json={"text": "accept"}
    )
    assert r.status_code == 201 and len(spawned) == 2

    # Stop without a running turn says so instead of pretending.
    r = api.post("/api/projects/titanic/experiments/%s/stop" % exp_id)
    assert r.json()["stopped"] is False

    assert api.get("/api/projects/titanic/experiments/ghost-1").status_code == 404
