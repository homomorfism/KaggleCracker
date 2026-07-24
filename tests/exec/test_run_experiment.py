"""One test per exec fixture, plus a schema-rejection and a precheck test.

Every test drives the real dispatch() path, so validation runs before the body
and the precheck before the (here absent) gate, exactly as production does. Each
asserts on a consequence: the exact error kind, its retryable flag, and — where
it matters — that a stray number never leaked or a file never changed.

The workspace is redirected to tmp_path via the `workspace` fixture, so no test
writes into the real workspace/ (see the KC_WORKSPACE rule in AGENTS.md).
"""

from dataclasses import dataclass
from pathlib import Path

import pytest

from core.loop import RunState, dispatch
from core.registry import Registry
from tools.exec.experiments import RECORD_RESULT, RUN_EXPERIMENT

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@dataclass
class Call:
    name: str
    args: dict


def _mute(*a, **k):
    return None


def _fixture(name):
    """Source text of a tests/fixtures script, handed to run_experiment as its
    `code` argument (run_experiment writes it out and executes it)."""
    return (FIXTURES / name).read_text()


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Point KC_WORKSPACE at a fresh tmp dir and seed one dataset so
    run_experiment's existence check passes. Returns the workspace root."""
    monkeypatch.setenv("KC_WORKSPACE", str(tmp_path))
    data = tmp_path / "data"
    data.mkdir(parents=True)
    (data / "train.csv").write_text("id,label\n1,0\n2,1\n")
    return tmp_path


def _run(code, **overrides):
    """Dispatch one run_experiment call. dataset_ref defaults to the seeded
    train.csv; overrides let a test set timeout_s and friends. Going through
    dispatch means validate() fills the timeout_s/cv_folds defaults for us."""
    args = {"code": code, "dataset_ref": "train.csv"}
    args.update(overrides)
    reg = Registry()
    reg.register(RUN_EXPERIMENT)
    return dispatch(Call("run_experiment", args), reg, RunState(),
                    input_fn=_mute, output_fn=_mute)


def _record_call(args):
    reg = Registry()
    reg.register(RECORD_RESULT)
    return dispatch(Call("record_experiment_result", args), reg, RunState(),
                    input_fn=_mute, output_fn=_mute)


# --- one test per fixture: exact kind AND retryable flag --------------------


def test_raises_is_exec_failed_and_carries_the_valueerror_text(workspace):
    result = _run(_fixture("raises.py"))
    assert result["ok"] is False
    assert result["error"]["kind"] == "exec_failed"
    assert result["error"]["retryable"] is False
    # The crash reason must survive into the message, not be swallowed.
    assert "ValueError" in result["error"]["msg"]
    assert "boom_from_raises" in result["error"]["msg"]


def test_hangs_is_timeout_and_not_retryable(workspace):
    # timeout_s=2 keeps the test fast; the identical call would hang again, so
    # the tool marks the timeout non-retryable.
    result = _run(_fixture("hangs.py"), timeout_s=2)
    assert result["ok"] is False
    assert result["error"]["kind"] == "timeout"
    assert result["error"]["retryable"] is False


def test_no_score_is_bad_input_and_leaks_no_stray_number(workspace):
    result = _run(_fixture("no_score.py"))
    assert result["ok"] is False
    assert result["error"]["kind"] == "bad_input"
    assert result["error"]["retryable"] is True
    # "mean auc = 0.8814" is printed for something else; it must never be
    # mistaken for the score or echoed back anywhere in the returned envelope.
    assert "0.8814" not in repr(result)


def test_bad_score_is_bad_input_and_retryable(workspace):
    result = _run(_fixture("bad_score.py"))
    assert result["ok"] is False
    assert result["error"]["kind"] == "bad_input"
    assert result["error"]["retryable"] is True


def test_good_returns_ok_with_the_parsed_score(workspace):
    result = _run(_fixture("good.py"))
    assert result["ok"] is True
    assert result["data"]["cv_score"] == 0.8814


# --- schema rejection: a bad arg stops before the body ----------------------


def test_oversized_timeout_is_rejected_before_any_script_is_written(workspace):
    # timeout_s max is 1800; 99999 must be a bad_input from validate(), and the
    # body must never run -> no experiment script is staged on disk.
    result = _run(_fixture("good.py"), timeout_s=99999)
    assert result["ok"] is False
    assert result["error"]["kind"] == "bad_input"
    exp_dir = workspace / "experiments"
    assert not exp_dir.exists() or list(exp_dir.glob("*.py")) == []


# --- precheck: an impossible fold count stops before the write --------------


def test_too_many_fold_scores_rejected_and_leaderboard_untouched(workspace):
    leaderboard = workspace / "experiments" / "leaderboard.jsonl"
    leaderboard.parent.mkdir(parents=True)
    seeded = '{"experiment_id": "seed", "cv_score": 0.5, "fold_scores": [], "notes": ""}\n'
    leaderboard.write_text(seeded)

    result = _record_call({
        "experiment_id": "e1",
        "cv_score": 0.5,
        "fold_scores": [0.5] * 21,  # more folds than _MAX_CV_FOLDS (20)
        "notes": "",
    })
    assert result["ok"] is False
    assert result["error"]["kind"] == "bad_input"
    # The rejected call must not have appended: the file is byte-for-byte the same.
    assert leaderboard.read_text() == seeded
