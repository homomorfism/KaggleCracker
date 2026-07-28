"""A shell for the coding agents: inspect the workspace, install packages.

Commands run with cwd pinned to the workspace and the project venv's bin dir
first on PATH, so `python` and `pip` resolve to the same interpreter the
experiment/EDA scripts run under — `pip install lightgbm` in this tool is
immediately importable in the next run_experiment script. Network access is
whatever the machine has; this is a local personal workbench, not a hosted
sandbox, and the human asked for exactly that.

Ungated on purpose: the projects live under the workspace and every artifact
the agent produces is reviewable. The cap that matters is the timeout.
"""

import os
import subprocess
import sys
from pathlib import Path

from core.contracts import err, ok
from core.registry import ToolSpec


def _workspace_root():
    return Path(os.environ.get("KC_WORKSPACE", "workspace"))


def _tail(text, limit=2000):
    return (text or "").strip()[-limit:]


def run_bash(args):
    command = args["command"]
    timeout_s = args["timeout_s"]

    ws = _workspace_root()
    env = dict(os.environ)
    venv_bin = Path(sys.executable).parent
    # ensurepip creates pip3 but not pip; without this, a bare `pip install`
    # silently resolves to some global pip and installs into the wrong
    # environment. Self-heal so the venv wins for both spellings.
    if not (venv_bin / "pip").exists() and (venv_bin / "pip3").exists():
        try:
            (venv_bin / "pip").symlink_to(venv_bin / "pip3")
        except OSError:
            pass  # PATH still prefers the venv for python/pip3
    env["PATH"] = str(venv_bin) + os.pathsep + env.get("PATH", "")

    try:
        proc = subprocess.run(
            ["/bin/bash", "-c", command],
            cwd=str(ws),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return err(
            "timeout",
            "command exceeded %ss; raise timeout_s or do less in one call" % timeout_s,
            retryable=False,
        )
    except OSError as e:
        return err("exec_failed", "could not start shell: %s" % e)

    if proc.returncode != 0:
        # The command's own words are the useful part of a failure.
        return err(
            "exec_failed",
            "exit %d: %s" % (proc.returncode, _tail(proc.stderr) or _tail(proc.stdout)),
            retryable=True,
        )
    return ok(stdout_tail=_tail(proc.stdout), stderr_tail=_tail(proc.stderr))


RUN_BASH = ToolSpec(
    name="run_bash",
    description=(
        "Run one bash command in the workspace (network available; the project "
        "venv's python/pip are first on PATH). Call it to inspect files (ls, "
        "head), check the environment, or install a missing package with "
        "'pip install <name>' — packages installed here import cleanly in later "
        "run_experiment / run_eda_script calls. Do NOT use it to run training "
        "or analysis code — run_experiment and run_eda_script exist for that "
        "and capture their outputs properly. Do NOT start long-lived processes "
        "(servers, watchers); the command is killed at timeout_s."
    ),
    parameters={
        "command": {"type": "str", "required": True},
        "timeout_s": {"type": "int", "min": 1, "max": 600, "default": 120},
    },
    fn=run_bash,
)


def register(registry):
    """Register the shell tool into the given registry."""
    registry.register(RUN_BASH)
    return registry
