"""The demo run end to end, in process: real recon tools on a real (tiny) CSV,
FakeModel prose, everything mirrored into the journal."""

import os

import pytest

from ui import projects
from ui.journal import read_events, run_status
from ui.run_analysis import run_demo_analysis, run_live_analysis, wait_for_gate_answer

_TRAIN = (
    "id,x1,x2,y\n"
    "1,0.5,10,fit\n"
    "2,0.7,12,fit\n"
    "3,0.9,14,sick\n"
    "4,0.4,9,fit\n"
    "5,1.1,16,sick\n"
    "6,0.6,11,fit\n"
    "7,1.3,18,sick\n"
    "8,0.5,10,fit\n"
)
_TEST = "id,x1,x2\n9,0.6,11\n10,1.2,17\n"


def _demo_project():
    projects.create_project("Demo", "predict y from x1/x2", "y")
    projects.save_data_file("demo", "train.csv", _TRAIN.encode())
    projects.save_data_file("demo", "test.csv", _TEST.encode())


def _run():
    run_id = projects.new_run_id("demo")
    run_demo_analysis("demo", run_id, pace=0)
    journal_path = projects.run_dir("demo", run_id) / "journal.jsonl"
    return run_id, read_events(journal_path)


def test_demo_run_journals_the_whole_loop(ui_projects_tmp):
    _demo_project()
    workspace_before = os.environ.get("KC_WORKSPACE")
    run_id, events = _run()
    types = [e["type"] for e in events]

    assert types[0] == "run_started"
    assert types[-1] == "run_finished"
    assert "reason" in types and "act" in types

    # Real numbers from the real file: the profile result must carry the checks
    # the script asked for, parsed back into structured data (not raw text).
    profile_ok = [
        e for e in events if e["type"] == "tool_ok" and e["tool"] == "profile_dataset"
    ]
    assert profile_ok, "no successful profile_dataset result reached the journal"
    assert "missingness" in profile_ok[0]["data"]
    assert profile_ok[0]["data"]["rows_profiled"] == 8

    # The deliberate original.csv probe: the error reaches the journal as its
    # own branch, with the closed-list kind, not as valid data.
    errors = [e for e in events if e["type"] == "tool_error"]
    assert any(e["kind"] == "not_found" for e in errors)

    # Consequences on disk: the plan was written into the project workspace and
    # the report landed next to the journal.
    assert (projects.project_dir("demo") / "plans" / "train.csv.plan.md").is_file()
    assert (projects.run_dir("demo", run_id) / "report.md").is_file()

    # The runner restored the environment it borrowed.
    assert os.environ.get("KC_WORKSPACE") == workspace_before


def test_second_run_goes_through_the_gate(ui_projects_tmp):
    _demo_project()
    _run()  # first run writes the plan ungated
    _, events = _run()  # second run must overwrite, which is gated

    prompts = [e for e in events if e["type"] == "gate_prompt"]
    answers = [e for e in events if e["type"] == "gate_answer"]
    assert prompts and answers
    # The journal never passes a scripted approval off as a human one.
    assert answers[0]["scripted"] is True
    assert any(
        e["type"] == "tool_ok" and e["tool"] == "overwrite_preprocessing_plan"
        for e in events
    )
    assert (projects.project_dir("demo") / "plans" / "train.csv.plan.md").is_file()


def test_gate_answer_file_is_read_once_and_consumed(tmp_path):
    (tmp_path / "gate_answer.txt").write_text("yes\n")
    answer_fn = wait_for_gate_answer(tmp_path, timeout=1, poll=0.01)
    assert answer_fn() == "yes"
    # Consumed: a second gate in the same run must get its own fresh answer.
    assert not (tmp_path / "gate_answer.txt").exists()


def test_gate_timeout_is_a_denial(tmp_path):
    answer_fn = wait_for_gate_answer(tmp_path, timeout=0.05, poll=0.01)
    # No file ever appears: the empty string is a denial by the gate's rules.
    assert answer_fn() == ""


def test_live_run_without_api_key_fails_into_the_journal(ui_projects_tmp, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _demo_project()
    run_id = projects.new_run_id("demo")
    with pytest.raises(ValueError):
        run_live_analysis("demo", run_id)
    journal_path = projects.run_dir("demo", run_id) / "journal.jsonl"
    assert run_status(journal_path) == "failed"


def test_run_without_data_fails_into_the_journal(ui_projects_tmp):
    projects.create_project("Empty")
    run_id = projects.new_run_id("empty")
    with pytest.raises(ValueError):
        run_demo_analysis("empty", run_id, pace=0)
    journal_path = projects.run_dir("empty", run_id) / "journal.jsonl"
    assert run_status(journal_path) == "failed"
