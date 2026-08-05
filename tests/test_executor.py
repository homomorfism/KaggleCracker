"""Executor behaviour.

The flag assertions are the standing evidence for success criterion 5 — "no
experiment container ever had network access" — and they run without Docker.
The `live` tests actually start containers and are excluded from the default
suite.
"""

from __future__ import annotations

import pytest

from kagglecracker.executors.base import RunStatus
from kagglecracker.executors.fake import FakeExecutor, FakeOutcome
from kagglecracker.executors.local_docker import LocalDockerExecutor

# --------------------------------------------------------------------------
# run flags — the security posture, asserted without starting a container
# --------------------------------------------------------------------------


@pytest.fixture
def flags(settings, tmp_path):
    return LocalDockerExecutor(settings).build_flags(
        run_dir=tmp_path / "run", container_name="kc-test"
    )


def _flag_value(flags: list[str], name: str) -> str:
    return flags[flags.index(name) + 1]


def test_network_is_disabled(flags):
    assert _flag_value(flags, "--network") == "none"


def test_all_capabilities_are_dropped(flags):
    assert _flag_value(flags, "--cap-drop") == "ALL"


def test_privilege_escalation_is_blocked(flags):
    assert _flag_value(flags, "--security-opt") == "no-new-privileges"


def test_data_is_mounted_read_only(flags):
    mounts = [flags[i + 1] for i, f in enumerate(flags) if f == "-v"]
    data_mount = next(m for m in mounts if ":/data" in m)
    assert data_mount.endswith(":ro"), data_mount


def test_workspace_is_writable(flags):
    mounts = [flags[i + 1] for i, f in enumerate(flags) if f == "-v"]
    assert any(m.endswith("/workspace:rw") for m in mounts)


def test_resource_limits_are_set(flags, settings):
    assert _flag_value(flags, "--cpus") == str(settings.docker_cpus)
    assert _flag_value(flags, "--memory") == settings.docker_memory
    assert _flag_value(flags, "--pids-limit") == str(settings.docker_pids_limit)


def test_container_runs_as_the_host_user(flags):
    """Otherwise writes to the bind-mounted workspace fail with a permission error
    that reads exactly like a bug in the generated code."""
    import os

    assert _flag_value(flags, "--user") == f"{os.getuid()}:{os.getgid()}"


def test_had_network_reads_the_recorded_flags(settings, tmp_path):
    from kagglecracker.executors.base import RunResult

    hardened = RunResult(
        status=RunStatus.OK, exit_code=0, duration_s=1.0, workspace=tmp_path,
        run_flags=["--network", "none"],
    )
    assert not hardened.had_network

    leaky = RunResult(
        status=RunStatus.OK, exit_code=0, duration_s=1.0, workspace=tmp_path, run_flags=[],
    )
    assert leaky.had_network


# --------------------------------------------------------------------------
# status semantics
# --------------------------------------------------------------------------


def test_infra_error_is_not_the_agents_fault():
    """It must never reach the debug action — the code never ran."""
    assert not RunStatus.INFRA_ERROR.is_agent_fault
    assert RunStatus.ERROR.is_agent_fault
    assert RunStatus.TIMEOUT.is_agent_fault


def test_timeout_carries_its_own_message_because_there_is_no_traceback(tmp_path):
    """A SIGKILLed process leaves nothing to quote. If the status did not carry
    its own text, the debug prompt would be handed an empty string."""
    fake = FakeExecutor(root=tmp_path, scripted=[FakeOutcome.timeout(900)])
    result = fake.run("print(1)", timeout_s=900, node_id="n1")
    assert result.status is RunStatus.TIMEOUT
    assert result.error_text
    assert "900" in result.error_text


# --------------------------------------------------------------------------
# FakeExecutor
# --------------------------------------------------------------------------


def test_fake_writes_output_files_only_on_success(tmp_path):
    fake = FakeExecutor(
        root=tmp_path,
        scripted=[
            FakeOutcome.ok(0.81, [0.8, 0.82], "PassengerId,Transported\n0013_01,True\n"),
            FakeOutcome.crash("ValueError: boom"),
        ],
    )
    good = fake.run("code", timeout_s=10, node_id="n1")
    assert (good.workspace / "metrics.json").exists()
    assert (good.workspace / "submission.csv").exists()

    bad = fake.run("code", timeout_s=10, node_id="n2")
    assert bad.status is RunStatus.ERROR
    assert not (bad.workspace / "metrics.json").exists()


def test_fake_falls_back_to_default_when_script_is_exhausted(tmp_path):
    fake = FakeExecutor(root=tmp_path, scripted=[FakeOutcome.crash("boom")])
    assert fake.run("c", timeout_s=1, node_id="n1").status is RunStatus.ERROR
    assert fake.run("c", timeout_s=1, node_id="n2").status is RunStatus.OK


def test_fake_records_calls(tmp_path):
    fake = FakeExecutor(root=tmp_path)
    fake.run("print(1)", timeout_s=42, node_id="n1")
    assert fake.calls == [("n1", "print(1)", 42)]


def test_fake_reports_the_same_hardened_flags(tmp_path):
    result = FakeExecutor(root=tmp_path).run("c", timeout_s=1, node_id="n1")
    assert not result.had_network


# --------------------------------------------------------------------------
# live — real containers. `pytest -m live`
# --------------------------------------------------------------------------


@pytest.mark.live
def test_live_successful_run(settings, tmp_path):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    code = "from pathlib import Path\nPath('ok.txt').write_text('hi')\nprint('done')\n"
    result = LocalDockerExecutor(settings).run(code, timeout_s=120, node_id="live-ok")
    assert result.status is RunStatus.OK, result.error_text
    assert (result.workspace / "ok.txt").read_text() == "hi"
    assert "done" in result.stdout_tail


@pytest.mark.live
def test_live_crash_captures_traceback(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    result = LocalDockerExecutor(settings).run(
        "raise ValueError('deliberate')", timeout_s=120, node_id="live-crash"
    )
    assert result.status is RunStatus.ERROR
    assert result.exit_code != 0
    assert "ValueError: deliberate" in result.error_text


@pytest.mark.live
def test_live_timeout_kills_the_container(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    result = LocalDockerExecutor(settings).run(
        "import time\ntime.sleep(600)", timeout_s=10, node_id="live-timeout"
    )
    assert result.status is RunStatus.TIMEOUT
    assert result.exit_code is None
    assert "traceback" in result.error_text.lower()


@pytest.mark.live
def test_live_container_has_no_network(settings):
    """Criterion 5, proven rather than asserted from flags."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    code = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=5)\n"
        "    print('NETWORK REACHABLE')\n"
        "except OSError as e:\n"
        "    print('no network:', type(e).__name__)\n"
    )
    result = LocalDockerExecutor(settings).run(code, timeout_s=120, node_id="live-net")
    assert "NETWORK REACHABLE" not in result.stdout_tail
    assert "no network" in result.stdout_tail


@pytest.mark.live
def test_live_data_mount_is_read_only(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "train.csv").write_text("a,b\n1,2\n")
    code = (
        "try:\n"
        "    open('/data/train.csv', 'w').write('clobbered')\n"
        "    print('WROTE TO /data')\n"
        "except OSError as e:\n"
        "    print('read-only:', type(e).__name__)\n"
    )
    result = LocalDockerExecutor(settings).run(code, timeout_s=120, node_id="live-ro")
    assert "WROTE TO /data" not in result.stdout_tail
    assert (settings.data_dir / "train.csv").read_text() == "a,b\n1,2\n"
