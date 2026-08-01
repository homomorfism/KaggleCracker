"""Runs agent-written code in a local Docker container.

    code string
        │
        ▼
    run_dir/train.py                     (host, one directory per node)
        │
        ▼
    docker run --network none --cap-drop ALL --security-opt no-new-privileges
               --user <host uid:gid> --cpus N --memory M --pids-limit P
               -v data:/data:ro -v run_dir:/workspace:rw
        │
        ├── exits 0            -> OK
        ├── exits non-zero     -> ERROR      (stderr tail is the traceback)
        ├── still alive at T   -> docker kill, TIMEOUT, no traceback
        └── never started      -> INFRA_ERROR

The exact flag list is recorded on every RunResult. That record — not this
comment — is the evidence for "no experiment container ever had network access".
"""

from __future__ import annotations

import os
import subprocess
import time
import uuid
from pathlib import Path

from kagglecracker.config import Settings
from kagglecracker.executors.base import RunResult, RunStatus

TRAIN_FILENAME = "train.py"
STDOUT_FILENAME = "stdout.log"

# How much of stdout to carry in memory for the debug prompt. The full log is
# always on disk; a 200-line tail is enough to hold a traceback without letting
# a runaway print loop blow the model's context window.
STDOUT_TAIL_LINES = 200
STDOUT_TAIL_CHARS = 20_000

# Substrings that mean the container never really ran. These are environment
# failures, not defects in the generated code, so they must not reach the debug
# action — it would try to repair code that was never executed.
_INFRA_MARKERS = (
    "cannot connect to the docker daemon",
    "is the docker daemon running",
    "no such image",
    "unable to find image",
    "manifest unknown",
    "error response from daemon",
    "permission denied while trying to connect",
)


def _tail(text: str) -> str:
    lines = text.splitlines()[-STDOUT_TAIL_LINES:]
    tail = "\n".join(lines)
    if len(tail) > STDOUT_TAIL_CHARS:
        tail = "...(truncated)...\n" + tail[-STDOUT_TAIL_CHARS:]
    return tail


class LocalDockerExecutor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # -- flags ------------------------------------------------------------

    def build_flags(self, *, run_dir: Path, container_name: str) -> list[str]:
        s = self.settings
        return [
            "--name", container_name,
            "--rm",
            # The whole security posture, in five flags.
            "--network", "none",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--pids-limit", str(s.docker_pids_limit),
            # Match the host user so files written into the bind-mounted
            # workspace are owned by us rather than by root. Without this the
            # image's non-root `runner` (uid 1000) may not own the mount and
            # every write fails with a permission error that reads like a bug
            # in the generated code.
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--cpus", str(s.docker_cpus),
            "--memory", s.docker_memory,
            "-v", f"{s.data_dir.resolve()}:/data:ro",
            "-v", f"{run_dir.resolve()}:/workspace:rw",
            "-w", "/workspace",
        ]

    # -- run --------------------------------------------------------------

    def run(self, code: str, *, timeout_s: int, node_id: str) -> RunResult:
        run_dir = self.settings.runs_dir / node_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / TRAIN_FILENAME).write_text(code)
        # Pre-create the cache dir the image's env vars point at, so a library
        # reaching for it never fails on a missing path.
        (run_dir / ".cache").mkdir(exist_ok=True)

        container_name = f"kc-{node_id}-{uuid.uuid4().hex[:8]}"
        flags = self.build_flags(run_dir=run_dir, container_name=container_name)
        cmd = ["docker", "run", *flags, self.settings.docker_image,
               "python", TRAIN_FILENAME]

        stdout_path = run_dir / STDOUT_FILENAME
        started = time.monotonic()
        timed_out = False

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            output = proc.stdout + proc.stderr
            exit_code: int | None = proc.returncode
        except FileNotFoundError:
            return self._infra(run_dir, flags, time.monotonic() - started,
                               "docker is not installed or not on PATH")
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            output = (exc.stdout or "") + (exc.stderr or "")
            if isinstance(output, bytes):  # pragma: no cover - defensive
                output = output.decode("utf-8", "replace")
            exit_code = None
            # `docker run` returning control to us does NOT stop the container.
            # Without this kill it keeps burning the CPU we just said we only
            # have one of, and the next node contends with a ghost.
            subprocess.run(
                ["docker", "kill", container_name],
                capture_output=True, text=True, timeout=30, check=False,
            )

        duration = time.monotonic() - started
        stdout_path.write_text(output)

        if not timed_out and exit_code != 0:
            lowered = output.lower()
            if any(m in lowered for m in _INFRA_MARKERS):
                return self._infra(run_dir, flags, duration, _tail(output), stdout_path)

        if timed_out:
            return RunResult(
                status=RunStatus.TIMEOUT,
                exit_code=None,
                duration_s=duration,
                workspace=run_dir,
                stdout_path=stdout_path,
                stdout_tail=_tail(output),
                # A SIGKILLed process leaves no traceback. Say so explicitly, or
                # the debug prompt is handed an empty string and invents a bug.
                error_text=(
                    f"The script was killed at the {timeout_s}s wall-clock limit and produced "
                    f"no traceback. It did not finish writing its output files. Make it finish "
                    f"inside the budget: fewer folds' worth of work per second, a cheaper model, "
                    f"fewer estimators, or subsample during search."
                ),
                run_flags=flags,
            )

        if exit_code != 0:
            return RunResult(
                status=RunStatus.ERROR,
                exit_code=exit_code,
                duration_s=duration,
                workspace=run_dir,
                stdout_path=stdout_path,
                stdout_tail=_tail(output),
                error_text=_tail(output),
                run_flags=flags,
            )

        return RunResult(
            status=RunStatus.OK,
            exit_code=0,
            duration_s=duration,
            workspace=run_dir,
            stdout_path=stdout_path,
            stdout_tail=_tail(output),
            run_flags=flags,
        )

    def _infra(
        self,
        run_dir: Path,
        flags: list[str],
        duration: float,
        message: str,
        stdout_path: Path | None = None,
    ) -> RunResult:
        return RunResult(
            status=RunStatus.INFRA_ERROR,
            exit_code=None,
            duration_s=duration,
            workspace=run_dir,
            stdout_path=stdout_path,
            error_text=message,
            run_flags=flags,
        )
