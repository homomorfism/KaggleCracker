"""The HTTP API end to end against the FastAPI app, and for the run test a
real subprocess writing a real journal."""

import json
import time

import pytest
from fastapi.testclient import TestClient

import ui.app
from ui import projects

_CSV = "id,x1,y\n1,0.5,fit\n2,0.9,sick\n3,0.4,fit\n4,1.2,sick\n"


@pytest.fixture
def api():
    # raise_server_exceptions=False so unmapped exceptions surface as the 500s
    # a browser would see, not as test-process tracebacks.
    return TestClient(ui.app.app, raise_server_exceptions=False)


def test_project_lifecycle_over_http(api):
    r = api.post(
        "/api/projects",
        json={"name": "S6E7 Health", "description": "predict", "target": "y"},
    )
    assert r.status_code == 201 and r.json()["slug"] == "s6e7-health"

    r = api.post("/api/projects/s6e7-health/files?name=train.csv", content=_CSV.encode())
    assert r.status_code == 201 and r.json()["bytes"] == len(_CSV)

    r = api.get("/api/projects")
    assert r.status_code == 200
    listing = r.json()
    assert [p["slug"] for p in listing["projects"]] == ["s6e7-health"]
    assert listing["projects"][0]["files"][0]["name"] == "train.csv"


def test_bad_requests_map_to_400_and_404(api):
    r = api.post("/api/projects", json={"name": "###"})
    assert r.status_code == 400 and "error" in r.json()

    r = api.get("/api/projects/ghost")
    assert r.status_code == 404 and "error" in r.json()

    # Evil filename is rejected before anything is written (400, not 500).
    r = api.post("/api/projects/ghost/files?name=../evil.csv", content=b"a\n1\n")
    assert r.status_code == 400

    # Starting a run on a project with no data is the user's mistake, said plainly.
    api.post("/api/projects", json={"name": "bare"})
    r = api.post("/api/projects/bare/runs", json={})
    assert r.status_code == 400 and "no data files" in r.json()["error"]


def test_gate_endpoint_delivers_the_answer_file(api):
    api.post("/api/projects", json={"name": "gated"})
    run_id = projects.new_run_id("gated")

    r = api.post("/api/projects/gated/runs/%s/gate" % run_id, json={"answer": "yes"})
    assert r.status_code == 200
    # The consequence: the file the live runner polls for, atomically complete.
    assert (projects.run_dir("gated", run_id) / "gate_answer.txt").read_text() == "yes"

    r = api.post("/api/projects/gated/runs/%s/gate" % run_id, json={"answer": "x" * 200})
    assert r.status_code == 400
    r = api.post(
        "/api/projects/gated/runs/run-00000000-000000/gate", json={"answer": "yes"}
    )
    assert r.status_code == 404


def test_capabilities_reflect_the_server_environment(api, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = api.get("/api/capabilities")
    assert r.status_code == 200 and r.json()["live"] is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert api.get("/api/capabilities").json()["live"] is True


def test_live_mode_rejected_without_api_key(api, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    api.post("/api/projects", json={"name": "lv"})
    api.post("/api/projects/lv/files?name=train.csv", content=_CSV.encode())
    r = api.post("/api/projects/lv/runs", json={"mode": "live"})
    assert r.status_code == 400 and "ANTHROPIC_API_KEY" in r.json()["error"]
    # Consequence: the rejection happened before a run was claimed.
    assert api.get("/api/projects/lv").json()["runs"] == []


def test_plans_and_findings_routes(api):
    api.post("/api/projects", json={"name": "prep"})
    plan = projects.project_dir("prep") / "plans" / "train.csv.plan.md"
    plan.write_text("# Preprocessing plan — train.csv\n\nimpute the medians.")

    r = api.get("/api/projects/prep/plans")
    assert r.status_code == 200
    body = r.json()
    assert body["plans"][0]["dataset"] == "train.csv"
    assert "impute" in body["plans"][0]["markdown"]

    r = api.get("/api/projects/prep/findings")
    assert r.status_code == 200 and r.json()["findings"] == []


def test_create_from_competition_url_spawns_setup(api, monkeypatch):
    spawned = []
    monkeypatch.setattr(ui.app, "_spawn", lambda mod, args, log: spawned.append((mod, args)))

    r = api.post(
        "/api/projects",
        json={"competition_url": "https://www.kaggle.com/competitions/titanic"},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["slug"] == "titanic"
    assert body["competition"] == "titanic"
    assert body["setup_state"] == "pending"
    assert spawned == [("ui.run_setup", ["titanic"])]

    # Bad URLs are the caller's mistake, before any project exists.
    r = api.post("/api/projects", json={"competition_url": "https://example.com/nope"})
    assert r.status_code == 400 and r.json()["kind"] == "bad_url"
    assert "nope" not in [p["slug"] for p in api.get("/api/projects").json()["projects"]]


def test_setup_status_and_retry(api, monkeypatch):
    spawned = []
    monkeypatch.setattr(ui.app, "_spawn", lambda mod, args, log: spawned.append(mod))
    api.post(
        "/api/projects",
        json={"competition_url": "https://www.kaggle.com/c/titanic"},
    )

    assert api.get("/api/projects/titanic/setup").json() == {"state": "none"}

    status_dir = projects.project_dir("titanic") / "setup"
    status_dir.mkdir(exist_ok=True)
    (status_dir / "status.json").write_text(json.dumps({"state": "downloading"}))
    assert api.get("/api/projects/titanic/setup").json()["state"] == "downloading"

    # A held lock means a setup is mid-flight: retry must not double-spawn.
    (status_dir / "lock").mkdir()
    r = api.post("/api/projects/titanic/setup/retry")
    assert r.status_code == 409
    (status_dir / "lock").rmdir()
    r = api.post("/api/projects/titanic/setup/retry")
    assert r.status_code == 200 and spawned == ["ui.run_setup", "ui.run_setup"]

    # A project without a linked competition has nothing to retry.
    api.post("/api/projects", json={"name": "plain"})
    assert api.post("/api/projects/plain/setup/retry").status_code == 400


def test_leaderboard_route_sorts_best_first(api):
    api.post("/api/projects", json={"name": "lb"})
    d = projects.project_dir("lb") / "experiments"
    d.mkdir()
    rows = [
        {"experiment_id": "a", "cv_score": 0.71},
        {"experiment_id": "b", "cv_score": 0.83},
    ]
    (d / "leaderboard.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    body = api.get("/api/projects/lb/leaderboard").json()
    assert [e["experiment_id"] for e in body["entries"]] == ["b", "a"]

    api.post("/api/projects", json={"name": "empty"})
    assert api.get("/api/projects/empty/leaderboard").json() == {"entries": []}


def test_root_serves_frontend_shell_or_build_instructions(api):
    # Works with or without a built dist: either the app shell or the
    # how-to-build text, never a 404 that strands the user.
    r = api.get("/")
    assert r.status_code == 200
    assert "KaggleCracker" in r.text or "kagglecracker" in r.text


def test_static_paths_cannot_escape_dist(api):
    r = api.get("/%2e%2e/CLAUDE.md")
    assert r.status_code == 200
    # Never the real file: traversal collapses to the app shell / instructions.
    assert "graded university assignment" not in r.text


def test_run_lifecycle_over_http(api):
    api.post("/api/projects", json={"name": "demo", "target": "y"})
    api.post("/api/projects/demo/files?name=train.csv", content=_CSV.encode())

    r = api.post("/api/projects/demo/runs", json={"pace": 0})
    assert r.status_code == 201
    run_id = r.json()["run_id"]

    # Poll exactly like the frontend will: events?since=N until a terminal status.
    deadline = time.time() + 30
    seen, run_state = [], "running"
    while time.time() < deadline and run_state == "running":
        r = api.get(
            "/api/projects/demo/runs/%s/events?since=%d"
            % (run_id, seen[-1]["seq"] if seen else 0)
        )
        assert r.status_code == 200
        payload = r.json()
        seen.extend(payload["events"])
        run_state = payload["status"]
        time.sleep(0.2)

    assert run_state == "finished", [e["type"] for e in seen]
    types = [e["type"] for e in seen]
    assert types[0] == "run_started" and "tool_ok" in types

    r = api.get("/api/projects/demo/runs/%s/report" % run_id)
    assert r.status_code == 200 and "reconnaissance report" in r.text
