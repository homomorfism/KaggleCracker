"""Wiring and behaviour of the submit tool, all driven through dispatch().

The shape tests pin that submit_to_kaggle is an irreversible, gated spec with a
precheck and the safe dry_run default. The behaviour tests then exercise the
real dispatch order — validate, precheck, gate, body — and assert on the
consequence each time: which answers approve, whether a quota was spent, and
what the human is shown at the gate.

The workspace is redirected to tmp_path via `submit_ws`, so no test touches the
real workspace/ (see the KC_WORKSPACE rule in AGENTS.md).
"""

import json
from dataclasses import dataclass
from datetime import date

import pytest

from core.loop import RunState, dispatch
from core.registry import Registry
from tools.exec.submit import DAILY_QUOTA, SUBMIT_TO_KAGGLE, register


# --- registration / shape ---------------------------------------------------


def _registered():
    """Register the tool into a fresh registry and hand back the stored spec."""
    return register(Registry()).get("submit_to_kaggle")


def test_submit_tool_registers_under_its_action_name():
    assert _registered() is SUBMIT_TO_KAGGLE


def test_submit_is_irreversible_and_carries_a_preview():
    # irreversible=True is what sends this tool through the gate; the registry
    # refuses to register an irreversible tool with no preview, so both must hold.
    spec = _registered()
    assert spec.irreversible is True
    assert spec.preview is not None


def test_submit_has_a_precheck_that_can_run_before_the_gate():
    assert _registered().precheck is not None


def test_dry_run_defaults_to_true():
    # An omitted flag must be a dry run, never a live submission.
    assert SUBMIT_TO_KAGGLE.parameters["dry_run"]["default"] is True


def test_daily_quota_is_a_positive_int():
    # bool is a subclass of int, so exclude it explicitly.
    assert isinstance(DAILY_QUOTA, int) and not isinstance(DAILY_QUOTA, bool)
    assert DAILY_QUOTA > 0


# --- behaviour: helpers -----------------------------------------------------


@dataclass
class Call:
    name: str
    args: dict


def _mute(*a, **k):
    return None


def _answer(text):
    """An input_fn that always returns `text`, ignoring the prompt."""
    return lambda _p="": text


def _approve(_p=""):
    return "yes"


def _deny(_p=""):
    return "no"


def _never(_p=""):
    # Used where the precheck must reject before the gate: if the gate is ever
    # reached, this fires and the test fails loudly instead of silently passing.
    raise AssertionError("gate must not be reached")


@pytest.fixture
def submit_ws(tmp_path, monkeypatch):
    """Point KC_WORKSPACE at a fresh tmp dir and seed a submission that matches
    the sample exactly (same columns, same row count). Returns the workspace."""
    monkeypatch.setenv("KC_WORKSPACE", str(tmp_path))
    data = tmp_path / "data"
    data.mkdir(parents=True)
    (data / "sample_submission.csv").write_text("id,target\n1,0\n2,0\n3,0\n")
    (tmp_path / "submission.csv").write_text("id,target\n1,0.9\n2,0.1\n3,0.5\n")
    return tmp_path


def _args(**over):
    a = {"submission_path": "submission.csv", "message": "first submission", "cv_score": 0.8814}
    a.update(over)
    return a


def _submit(args, input_fn, output_fn=_mute):
    reg = Registry()
    reg.register(SUBMIT_TO_KAGGLE)
    return dispatch(Call("submit_to_kaggle", args), reg, RunState(),
                    input_fn=input_fn, output_fn=output_fn)


def _quota_file(ws):
    return ws / "submissions" / "quota.json"


def _read_quota(ws):
    return json.loads(_quota_file(ws).read_text())


# --- behaviour: the gate accepts only yes / y -------------------------------


@pytest.mark.parametrize("answer", ["yes", "y"])
def test_only_yes_and_y_approve(submit_ws, answer):
    result = _submit(_args(), input_fn=_answer(answer))
    assert result["ok"] is True  # approved -> the (dry-run) body ran


@pytest.mark.parametrize("answer", ["", "ok", "sure", "go ahead", "n", "no"])
def test_everything_else_denies(submit_ws, answer):
    result = _submit(_args(), input_fn=_answer(answer))
    assert result["ok"] is False
    assert result["error"]["kind"] == "denied_by_human"
    assert result["error"]["retryable"] is False


# --- behaviour: quota is spent only by an approved real submission ----------


def test_approved_real_submission_increments_quota(submit_ws):
    assert not _quota_file(submit_ws).exists()  # nothing spent yet
    result = _submit(_args(dry_run=False), input_fn=_approve)
    assert result["ok"] is True
    assert result["data"]["submitted"] is True
    assert _read_quota(submit_ws)[date.today().isoformat()] == 1


def test_denied_real_submission_is_denied_and_spends_nothing(submit_ws):
    result = _submit(_args(dry_run=False), input_fn=_deny)
    assert result["ok"] is False
    assert result["error"]["kind"] == "denied_by_human"
    assert result["error"]["retryable"] is False
    # Denial must leave no trace: the quota file was never written.
    assert not _quota_file(submit_ws).exists()


def test_dry_run_default_spends_nothing_even_when_approved(submit_ws):
    # No dry_run passed -> default True. Even an approval must not upload or spend.
    result = _submit(_args(), input_fn=_approve)
    assert result["ok"] is True
    assert result["data"]["dry_run"] is True
    assert result["data"]["submitted"] is False
    assert not _quota_file(submit_ws).exists()


def test_exhausted_quota_is_rejected_before_the_gate(submit_ws):
    # Today's count already at the cap: the precheck must reject it, so the gate
    # is never reached (_never would fire) and the body never runs.
    _quota_file(submit_ws).parent.mkdir(parents=True)
    _quota_file(submit_ws).write_text(json.dumps({date.today().isoformat(): DAILY_QUOTA}))
    result = _submit(_args(dry_run=False), input_fn=_never)
    assert result["ok"] is False
    assert result["error"]["kind"] == "quota_exhausted"


# --- behaviour: the preview shows the human what they are approving ----------


def test_preview_shows_rows_score_remaining_and_irreversibility(submit_ws):
    shown = []
    _submit(_args(cv_score=0.8814), input_fn=_approve, output_fn=shown.append)
    assert len(shown) == 1
    preview = shown[0]
    assert "3 data rows" in preview                     # row count
    assert "0.8814" in preview                          # cv_score of the model
    assert "5 of 5" in preview and "remaining" in preview  # remaining count
    assert "cannot be undone" in preview                # the irreversibility line


# --- behaviour: a bad file is rejected before the gate, not by a human -------


def test_wrong_row_count_is_bad_input_before_the_gate(submit_ws):
    # Two data rows against the sample's three: a string comparison rejects this,
    # so the human is never asked (_never would fire if the gate were reached).
    (submit_ws / "submission.csv").write_text("id,target\n1,0.9\n2,0.1\n")
    result = _submit(_args(), input_fn=_never)
    assert result["ok"] is False
    assert result["error"]["kind"] == "bad_input"


def test_wrong_columns_is_bad_input_before_the_gate(submit_ws):
    # Header 'pred' where the sample says 'target': again caught before the gate.
    (submit_ws / "submission.csv").write_text("id,pred\n1,0.9\n2,0.1\n3,0.5\n")
    result = _submit(_args(), input_fn=_never)
    assert result["ok"] is False
    assert result["error"]["kind"] == "bad_input"
