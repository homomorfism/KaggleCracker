"""run_bash: real subprocesses in a tmp workspace, every branch an envelope."""

import sys

import pytest

from tools.exec.shell import run_bash


@pytest.fixture(autouse=True)
def shell_workspace(tmp_path, monkeypatch):
    ws = tmp_path / "workspace"
    (ws / "data").mkdir(parents=True)
    monkeypatch.setenv("KC_WORKSPACE", str(ws))
    return ws


def _run(command, timeout_s=30):
    return run_bash({"command": command, "timeout_s": timeout_s})


def test_runs_in_the_workspace_with_venv_python_first(shell_workspace):
    (shell_workspace / "data" / "train.csv").write_text("a\n1\n")
    out = _run("pwd && ls data && which python")
    assert out["ok"], out
    stdout = out["data"]["stdout_tail"]
    assert str(shell_workspace) in stdout
    assert "train.csv" in stdout
    # `python` resolves to the interpreter the experiment scripts run under.
    assert sys.executable in stdout


def test_nonzero_exit_is_exec_failed_with_the_commands_words(shell_workspace):
    out = _run("echo diagnosis >&2; exit 3")
    assert not out["ok"]
    assert out["error"]["kind"] == "exec_failed"
    assert out["error"]["retryable"]
    assert "exit 3" in out["error"]["msg"] and "diagnosis" in out["error"]["msg"]


def test_timeout_maps_to_timeout_kind(shell_workspace):
    out = _run("sleep 5", timeout_s=1)
    assert not out["ok"] and out["error"]["kind"] == "timeout"
