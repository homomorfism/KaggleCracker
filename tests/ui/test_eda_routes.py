"""The EDA chat routes: message -> spawn, version-gated polling, 409 while a
turn is running."""

import json

import pytest
from fastapi.testclient import TestClient

import ui.app
from ui import projects
from ui.run_eda import append_message, read_messages


@pytest.fixture
def api():
    return TestClient(ui.app.app, raise_server_exceptions=False)


@pytest.fixture
def project():
    return projects.create_project("titanic", competition="titanic")


def test_empty_project_has_empty_eda_state(api, project):
    body = api.get("/api/projects/titanic/eda").json()
    assert body["dashboard"]["panels"] == []
    assert body["chat"] == []
    assert body["agent"] == {"status": "idle", "run_id": None}


def test_posting_a_message_appends_and_spawns(api, project, monkeypatch):
    spawned = []
    monkeypatch.setattr(ui.app, "_spawn", lambda mod, args, log: spawned.append((mod, args)))

    r = api.post("/api/projects/titanic/eda/messages", json={"text": "add a heatmap"})
    assert r.status_code == 201
    assert r.json()["agent"]["status"] == "running"
    assert spawned == [("ui.run_eda", ["titanic"])]
    messages = read_messages("titanic")
    assert len(messages) == 1
    assert messages[0]["role"] == "user" and messages[0]["text"] == "add a heatmap"

    # Empty and oversized messages never reach the agent.
    assert api.post("/api/projects/titanic/eda/messages", json={"text": "  "}).status_code == 400
    assert (
        api.post("/api/projects/titanic/eda/messages", json={"text": "x" * 5000}).status_code
        == 400
    )


def test_lock_makes_messages_409(api, project, monkeypatch):
    monkeypatch.setattr(ui.app, "_spawn", lambda mod, args, log: None)
    lock = projects.project_dir("titanic") / "eda" / "lock"
    lock.mkdir(parents=True)
    r = api.post("/api/projects/titanic/eda/messages", json={"text": "hi"})
    assert r.status_code == 409
    assert read_messages("titanic") == []  # nothing queued behind the 409
    # The lock also reports the agent as running to pollers.
    assert api.get("/api/projects/titanic/eda").json()["agent"]["status"] == "running"


def test_version_gate_skips_unchanged_payload(api, project):
    d = projects.project_dir("titanic") / "eda"
    d.mkdir(parents=True)
    (d / "dashboard.json").write_text(
        json.dumps({"version": 3, "updated": "t", "panels": []})
    )
    assert api.get("/api/projects/titanic/eda?version=3").json() == {
        "unchanged": True,
        "version": 3,
    }
    body = api.get("/api/projects/titanic/eda?version=2").json()
    assert body["dashboard"]["version"] == 3


def test_chat_survives_and_orders_messages(api, project):
    append_message("titanic", "user", "first")
    append_message("titanic", "assistant", "did it")
    chat = api.get("/api/projects/titanic/eda").json()["chat"]
    assert [(m["seq"], m["role"]) for m in chat] == [(1, "user"), (2, "assistant")]


def test_eda_events_route(api, project):
    rdir = projects.project_dir("titanic") / "eda" / "runs" / "turn-1"
    rdir.mkdir(parents=True)
    (rdir / "journal.jsonl").write_text(
        json.dumps({"seq": 1, "ts": 0, "type": "run_started", "prompt": "p"})
        + "\n"
        + json.dumps({"seq": 2, "ts": 1, "type": "run_finished", "text": "done"})
        + "\n"
    )
    body = api.get("/api/projects/titanic/eda/runs/turn-1/events?since=0").json()
    assert body["status"] == "finished"
    assert [e["type"] for e in body["events"]] == ["run_started", "run_finished"]
    assert api.get("/api/projects/titanic/eda/runs/ghost/events").status_code == 404
