"""The data-preparation demo run: plan decisions executed on the real files."""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import ui.app
from ui import projects
from ui.journal import run_status
from ui.run_analysis import run_demo_analysis
from ui.run_prep import run_prep

# id: unique per row -> dropped. x1: one text value among floats -> mixed,
# coerced. x2: two blanks -> imputed. y: the target, never touched.
_TRAIN = (
    "id,x1,x2,y\n"
    "1,0.5,10,fit\n"
    "2,0.7,,fit\n"
    "3,oops,12,sick\n"
    "4,0.9,,fit\n"
    "5,1.1,14,sick\n"
    "6,0.5,9,fit\n"
)
_TEST = "id,x1,x2\n7,0.6,11\n8,1.2,13\n"


def _project_with_plan():
    projects.create_project("Demo", "predict y", "y")
    projects.save_data_file("demo", "train.csv", _TRAIN.encode())
    projects.save_data_file("demo", "test.csv", _TEST.encode())
    # The recon demo records the findings and writes the plan prep executes.
    run_demo_analysis("demo", projects.new_run_id("demo"), pace=0)


def test_prep_executes_the_plan_decisions(ui_projects_tmp):
    _project_with_plan()
    run_prep("demo")

    prep = projects.project_dir("demo") / "prep"
    assert run_status(prep / "journal.jsonl") == "finished"

    out = pd.read_csv(prep / "train.prepared.csv")
    # Flagged as an identifier -> dropped.
    assert "id" not in out.columns
    # Mixed -> coerced to numeric; 'oops' became the train median, so no holes.
    assert out["x1"].dtype.kind == "f"
    assert not out["x1"].isna().any()
    # Missing values -> imputed from train's median, none left.
    assert not out["x2"].isna().any()
    # The target column is untouched.
    assert list(out["y"]) == ["fit", "fit", "sick", "fit", "sick", "fit"]

    # The same decisions applied to the test file, with TRAIN's fill values.
    out_test = pd.read_csv(prep / "test.prepared.csv")
    assert "id" not in out_test.columns

    manifest = json.loads((prep / "manifest.json").read_text())
    assert {f["name"] for f in manifest["files"]} == {
        "train.prepared.csv",
        "test.prepared.csv",
    }
    assert any(d["column"] == "id" for d in manifest["decisions"]["dropped"])
    # Re-runnable: the lock never outlives the run.
    assert not (prep / "lock").exists()


def test_prep_without_plan_fails_into_the_journal(ui_projects_tmp):
    projects.create_project("Bare")
    projects.save_data_file("bare", "train.csv", b"a,b\n1,2\n")
    with pytest.raises(ValueError):
        run_prep("bare")
    prep = projects.project_dir("bare") / "prep"
    assert run_status(prep / "journal.jsonl") == "failed"
    assert not (prep / "lock").exists()


@pytest.fixture
def api():
    return TestClient(ui.app.app, raise_server_exceptions=False)


def test_prep_routes_spawn_and_status(api, ui_projects_tmp, monkeypatch):
    spawned = []
    monkeypatch.setattr(ui.app, "_spawn", lambda mod, args, log: spawned.append((mod, args)))

    api.post("/api/projects", json={"name": "demo", "target": "y"})
    # No plan yet: refused before anything spawns.
    r = api.post("/api/projects/demo/prep/run")
    assert r.status_code == 400 and spawned == []
    assert api.get("/api/projects/demo/prep").json()["status"] == "none"

    (projects.project_dir("demo") / "plans" / "train.csv.plan.md").write_text("# plan")
    r = api.post("/api/projects/demo/prep/run")
    assert r.status_code == 200 and spawned == [("ui.run_prep", ["demo", "--pace", "0.6"])]

    # The spawn marker counts as running, and a second start is refused.
    assert api.get("/api/projects/demo/prep").json()["status"] == "running"
    assert api.post("/api/projects/demo/prep/run").status_code == 409


def test_prepared_file_download(api, ui_projects_tmp):
    projects.create_project("Demo")
    prep = projects.project_dir("demo") / "prep"
    prep.mkdir()
    (prep / "train.prepared.csv").write_text("a,b\n1,2\n")

    r = api.get("/api/projects/demo/prep/files/train.prepared.csv")
    assert r.status_code == 200 and "a,b" in r.text
    assert api.get("/api/projects/demo/prep/files/nope.csv").status_code == 404
    # A slash can never reach the handler ({name} matches one path segment);
    # what CAN arrive — dot-prefixed or non-csv names — is rejected as 400.
    assert api.get("/api/projects/demo/prep/files/.hidden.csv").status_code == 400
    assert api.get("/api/projects/demo/prep/files/journal.jsonl").status_code == 400
