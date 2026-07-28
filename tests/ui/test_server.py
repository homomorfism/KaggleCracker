"""The HTTP API end to end: a real ThreadingHTTPServer on an ephemeral port,
and for the run test a real subprocess writing a real journal."""

import http.client
import json
import threading
import time

import pytest

from ui import projects
from ui.server import make_server

_CSV = "id,x1,y\n1,0.5,fit\n2,0.9,sick\n3,0.4,fit\n4,1.2,sick\n"


@pytest.fixture
def api():
    server = make_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address
    server.shutdown()


def _request(addr, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection(addr[0], addr[1], timeout=10)
    conn.request(method, path, body=body, headers=headers or {})
    response = conn.getresponse()
    data = response.read()
    conn.close()
    return response.status, data


def _json_request(addr, method, path, obj=None):
    body = json.dumps(obj).encode() if obj is not None else None
    status, data = _request(addr, method, path, body, {"Content-Type": "application/json"})
    return status, (json.loads(data) if data else None)


def test_project_lifecycle_over_http(api):
    status, created = _json_request(
        api, "POST", "/api/projects",
        {"name": "S6E7 Health", "description": "predict", "target": "y"},
    )
    assert status == 201 and created["slug"] == "s6e7-health"

    status, saved = _request(
        api, "POST", "/api/projects/s6e7-health/files?name=train.csv", _CSV.encode()
    )
    assert status == 201 and json.loads(saved)["bytes"] == len(_CSV)

    status, listing = _json_request(api, "GET", "/api/projects")
    assert status == 200
    assert [p["slug"] for p in listing["projects"]] == ["s6e7-health"]
    assert listing["projects"][0]["files"][0]["name"] == "train.csv"


def test_bad_requests_map_to_400_and_404(api):
    status, body = _json_request(api, "POST", "/api/projects", {"name": "###"})
    assert status == 400 and "error" in body

    status, body = _json_request(api, "GET", "/api/projects/ghost")
    assert status == 404 and "error" in body

    # Evil filename is rejected before anything is written (400, not 500).
    status, _ = _request(
        api, "POST", "/api/projects/ghost/files?name=../evil.csv", b"a\n1\n"
    )
    assert status == 400

    # Starting a run on a project with no data is the user's mistake, said plainly.
    _json_request(api, "POST", "/api/projects", {"name": "bare"})
    status, body = _json_request(api, "POST", "/api/projects/bare/runs", {})
    assert status == 400 and "no data files" in body["error"]


def test_gate_endpoint_delivers_the_answer_file(api):
    _json_request(api, "POST", "/api/projects", {"name": "gated"})
    run_id = projects.new_run_id("gated")

    status, _ = _json_request(
        api, "POST", "/api/projects/gated/runs/%s/gate" % run_id, {"answer": "yes"}
    )
    assert status == 200
    # The consequence: the file the live runner polls for, atomically complete.
    assert (projects.run_dir("gated", run_id) / "gate_answer.txt").read_text() == "yes"

    status, _ = _json_request(
        api, "POST", "/api/projects/gated/runs/%s/gate" % run_id, {"answer": "x" * 200}
    )
    assert status == 400
    status, _ = _json_request(
        api, "POST", "/api/projects/gated/runs/run-00000000-000000/gate", {"answer": "yes"}
    )
    assert status == 404


def test_capabilities_reflect_the_server_environment(api, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    status, body = _json_request(api, "GET", "/api/capabilities")
    assert status == 200 and body["live"] is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _, body = _json_request(api, "GET", "/api/capabilities")
    assert body["live"] is True


def test_live_mode_rejected_without_api_key(api, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _json_request(api, "POST", "/api/projects", {"name": "lv"})
    _request(api, "POST", "/api/projects/lv/files?name=train.csv", _CSV.encode())
    status, body = _json_request(api, "POST", "/api/projects/lv/runs", {"mode": "live"})
    assert status == 400 and "ANTHROPIC_API_KEY" in body["error"]
    # Consequence: the rejection happened before a run was claimed.
    _, detail = _json_request(api, "GET", "/api/projects/lv")
    assert detail["runs"] == []


def test_plans_and_findings_routes(api):
    _json_request(api, "POST", "/api/projects", {"name": "prep"})
    plan = projects.project_dir("prep") / "plans" / "train.csv.plan.md"
    plan.write_text("# Preprocessing plan — train.csv\n\nimpute the medians.")

    status, body = _json_request(api, "GET", "/api/projects/prep/plans")
    assert status == 200
    assert body["plans"][0]["dataset"] == "train.csv"
    assert "impute" in body["plans"][0]["markdown"]

    status, body = _json_request(api, "GET", "/api/projects/prep/findings")
    assert status == 200 and body["findings"] == []


def test_root_serves_frontend_shell_or_build_instructions(api):
    # Works with or without a built dist: either the app shell or the
    # how-to-build text, never a 404 that strands the user.
    status, body = _request(api, "GET", "/")
    assert status == 200
    assert b"KaggleCracker" in body or b"kagglecracker" in body


def test_static_paths_cannot_escape_dist(api):
    status, body = _request(api, "GET", "/../CLAUDE.md")
    assert status == 200
    # Never the real file: traversal collapses to the app shell / instructions.
    assert b"graded university assignment" not in body


def test_run_lifecycle_over_http(api):
    _json_request(api, "POST", "/api/projects", {"name": "demo", "target": "y"})
    _request(api, "POST", "/api/projects/demo/files?name=train.csv", _CSV.encode())

    status, started = _json_request(api, "POST", "/api/projects/demo/runs", {"pace": 0})
    assert status == 201
    run_id = started["run_id"]

    # Poll exactly like the frontend will: events?since=N until a terminal status.
    deadline = time.time() + 30
    seen, run_state = [], "running"
    while time.time() < deadline and run_state == "running":
        status, payload = _json_request(
            api, "GET",
            "/api/projects/demo/runs/%s/events?since=%d"
            % (run_id, seen[-1]["seq"] if seen else 0),
        )
        assert status == 200
        seen.extend(payload["events"])
        run_state = payload["status"]
        time.sleep(0.2)

    assert run_state == "finished", [e["type"] for e in seen]
    types = [e["type"] for e in seen]
    assert types[0] == "run_started" and "tool_ok" in types

    status, report = _request(api, "GET", "/api/projects/demo/runs/%s/report" % run_id)
    assert status == 200 and b"reconnaissance report" in report
