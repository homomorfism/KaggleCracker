"""Tests for the admin path's boundary.

The denial tests assert the CONSEQUENCE, not just the returned status: after a
refused "dry_run off" the flag is still on and no state file exists, and after a
refused "stop" no stop was recorded. A boundary that returns "denied" while
still doing the work would pass a status-only test.
"""

import json

import pytest

from agents import admin
from core.registry import Registry
from tools.exec import experiments, submit

ADMIN = "admin-42"
STRANGER = "someone-else"


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """Tmp workspace (quota file) + tmp runs/hw3 (journal, state) + a configured
    admin id. Returns (workspace, runs/hw3)."""
    ws = tmp_path / "workspace"
    runs = tmp_path / "runs"
    monkeypatch.setenv("KC_WORKSPACE", str(ws))
    monkeypatch.setenv("KC_RUNS", str(runs))
    monkeypatch.setenv(admin.ADMIN_ID_ENV, ADMIN)
    return ws, runs / "hw3"


def _journal(hw3):
    path = hw3 / "admin_journal.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _seed_quota(ws, spent):
    d = ws / "submissions"
    d.mkdir(parents=True, exist_ok=True)
    (d / "quota.json").write_text(json.dumps({submit._today(): spent}))


# --- the authorized path -----------------------------------------------------


def test_quota_reports_what_is_left_today(dirs):
    ws, hw3 = dirs
    _seed_quota(ws, 3)

    obj = admin.handle_command(ADMIN, "quota", output_fn=lambda *a: None)

    assert obj["status"] == "ok"
    data = obj["envelope"]["data"]
    assert data["spent"] == 3
    assert data["remaining"] == submit.DAILY_QUOTA - 3
    assert data["daily_quota"] == submit.DAILY_QUOTA


def test_quota_with_no_submissions_yet(dirs):
    obj = admin.handle_command(ADMIN, "quota", output_fn=lambda *a: None)
    assert obj["status"] == "ok"
    assert obj["envelope"]["data"]["remaining"] == submit.DAILY_QUOTA


def test_dry_run_off_then_on_flips_the_recorded_flag(dirs):
    _ws, hw3 = dirs
    assert admin.dry_run_enabled() is True          # safe default with no file

    assert admin.handle_command(ADMIN, "dry_run off")["status"] == "ok"
    assert admin.dry_run_enabled() is False
    assert json.loads((hw3 / "submit_state.json").read_text())["dry_run"] is False

    assert admin.handle_command(ADMIN, "dry_run on")["status"] == "ok"
    assert admin.dry_run_enabled() is True          # fully undone: it is reversible


def test_dry_run_off_never_touches_the_submit_tool_default(dirs):
    admin.handle_command(ADMIN, "dry_run off")

    # The hard rule: dry_run=True stays the schema default on submit_to_kaggle,
    # and the tool stays irreversible so the human gate still runs. The admin
    # flag records a decision; it does not rewrite the tool.
    assert submit.SUBMIT_TO_KAGGLE.parameters["dry_run"]["default"] is True
    assert submit.SUBMIT_TO_KAGGLE.irreversible is True


def test_stop_records_only_the_named_turn(dirs):
    _ws, hw3 = dirs
    obj = admin.handle_command(ADMIN, "stop turn-7")

    assert obj["status"] == "ok"
    assert admin.stop_requested("turn-7") is True
    assert admin.stop_requested("turn-8") is False   # one turn, not a global halt
    assert json.loads((hw3 / "stop_requests.json").read_text()) == ["turn-7"]


def test_stopping_the_same_turn_twice_is_one_request(dirs):
    _ws, hw3 = dirs
    admin.handle_command(ADMIN, "stop turn-7")
    admin.handle_command(ADMIN, "stop turn-7")
    assert json.loads((hw3 / "stop_requests.json").read_text()) == ["turn-7"]


def test_allowed_commands_are_journaled_with_the_caller(dirs):
    _ws, hw3 = dirs
    admin.handle_command(ADMIN, "quota", output_fn=lambda *a: None)

    entries = _journal(hw3)
    assert len(entries) == 1
    assert entries[0]["caller_id"] == ADMIN
    assert entries[0]["command"] == "quota"
    assert entries[0]["decision"] == "allowed"


# --- the denial branch -------------------------------------------------------


def test_a_stranger_is_denied_and_nothing_happens(dirs):
    _ws, hw3 = dirs

    obj = admin.handle_command(STRANGER, "dry_run off")

    assert obj["status"] == "denied"
    assert obj["needs_approval"] is False
    assert obj["envelope"] is None
    # The consequence, not just the status: no flag was flipped and no file written.
    assert admin.dry_run_enabled() is True
    assert not (hw3 / "submit_state.json").exists()


def test_a_denied_command_is_journaled_with_the_rejected_id(dirs):
    _ws, hw3 = dirs
    admin.handle_command(STRANGER, "stop turn-7")

    entries = _journal(hw3)
    assert len(entries) == 1
    assert entries[0]["decision"] == "denied"
    assert entries[0]["caller_id"] == STRANGER    # the rejected id is on the record
    assert entries[0]["command"] == "stop turn-7"
    # And no stop was actually recorded.
    assert admin.stop_requested("turn-7") is False
    assert not (hw3 / "stop_requests.json").exists()


def test_an_unset_admin_id_authorizes_nobody(dirs, monkeypatch):
    monkeypatch.delenv(admin.ADMIN_ID_ENV)

    # Not even the id that WAS the admin a moment ago, and not an empty caller
    # id, which is what an unset env var would equal if we compared naively.
    for caller in (ADMIN, "", None):
        assert admin.handle_command(caller, "quota")["status"] == "denied"


def test_a_blank_admin_id_authorizes_nobody(dirs, monkeypatch):
    monkeypatch.setenv(admin.ADMIN_ID_ENV, "   ")
    assert admin.handle_command("   ", "quota")["status"] == "denied"


def test_a_near_miss_id_is_not_the_admin(dirs):
    # No case folding, no stripping, no prefix match.
    for caller in (ADMIN.upper(), " " + ADMIN, ADMIN + "x", ADMIN[:-1]):
        assert admin.handle_command(caller, "quota")["status"] == "denied"


# --- bad commands from a real admin -----------------------------------------


def test_an_unknown_command_is_an_error_not_a_denial(dirs):
    _ws, hw3 = dirs
    obj = admin.handle_command(ADMIN, "delete everything")

    assert obj["status"] == "error"      # the boundary held; the command is unknown
    assert "unknown admin command" in obj["result"]
    assert _journal(hw3)[0]["decision"] == "error"


def test_a_compound_command_is_not_two_actions(dirs):
    _ws, hw3 = dirs
    # The command string is data. A caller who appends another instruction gets
    # one unknown command, not a flag flip plus something else.
    obj = admin.handle_command(ADMIN, "dry_run off; stop turn-7")

    assert obj["status"] == "error"
    assert admin.dry_run_enabled() is True
    assert not (hw3 / "submit_state.json").exists()
    assert admin.stop_requested("turn-7") is False


def test_stop_without_a_turn_id_is_rejected_before_the_body_runs(dirs):
    _ws, hw3 = dirs
    obj = admin.handle_command(ADMIN, "stop")

    assert obj["status"] == "error"
    assert obj["envelope"]["error"]["kind"] == "bad_input"
    assert "requires a turn id" in obj["envelope"]["error"]["msg"]
    # The precheck ran before the body: no file was created at all.
    assert not (hw3 / "stop_requests.json").exists()


def test_a_corrupt_quota_file_is_an_error_branch_not_a_crash(dirs):
    ws, _hw3 = dirs
    d = ws / "submissions"
    d.mkdir(parents=True)
    (d / "quota.json").write_text("{not json")

    obj = admin.handle_command(ADMIN, "quota", output_fn=lambda *a: None)

    assert obj["status"] == "error"
    assert obj["envelope"]["error"]["kind"] == "exec_failed"


# --- the two registries are separate ----------------------------------------


def test_the_privileged_registry_holds_only_the_admin_tools():
    reg = admin.privileged_registry()
    names = [s["name"] for s in reg.schemas()]
    assert names == ["read_submission_quota", "set_submit_dry_run", "request_turn_stop"]
    # The admin path cannot submit. Neither registry is a superset of the other.
    assert reg.get("submit_to_kaggle") is None
    assert reg.get("run_experiment") is None


def test_the_ordinary_registry_cannot_reach_the_admin_tools():
    ordinary = Registry()
    experiments.register(ordinary)
    submit.register(ordinary)

    for name in ("read_submission_quota", "set_submit_dry_run", "request_turn_stop"):
        assert ordinary.get(name) is None


def test_no_admin_tool_is_gated():
    # Every admin command is reversible, so none of them may claim the human
    # gate. The single gate stays on submit_to_kaggle.
    for spec in (admin.READ_SUBMISSION_QUOTA, admin.SET_SUBMIT_DRY_RUN,
                 admin.REQUEST_TURN_STOP):
        assert spec.irreversible is False


def test_enabled_must_be_a_real_bool(dirs):
    # The constrained parameter, checked through the registry the same way an
    # ordinary tool's arguments are.
    from core.loop import RunState, dispatch
    from collections import namedtuple
    Call = namedtuple("Call", "name args")

    env = dispatch(Call("set_submit_dry_run", {"enabled": "off"}),
                   admin.privileged_registry(), RunState())

    assert env["ok"] is False
    assert env["error"]["kind"] == "bad_input"
