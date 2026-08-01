"""Executor interface and its result type.

The engine talks only to this interface, never to Docker. That is what makes an
SshDockerExecutor a later addition rather than a rewrite — and `FakeExecutor`
lets the whole search loop be tested in under a second with no Docker at all.

    Executor.run(code, timeout_s) -> RunResult
         │
         ├── LocalDockerExecutor   week 1
         ├── SshDockerExecutor     v2, remote GPU box
         └── FakeExecutor          tests

An executor reports only what it can actually observe — did the process run,
crash, get killed, or fail to start. It never opens metrics.json or
submission.csv; judging output is the contract's job, and keeping that boundary
means the executor stays swappable.

    ┌─────────────────────────────────────────────────────────────┐
    │ exit 0            -> OK           stdout captured           │
    │ exit != 0         -> ERROR        traceback in error_text   │
    │ SIGKILL @ timeout -> TIMEOUT      NO traceback exists       │
    │ daemon/image fail -> INFRA_ERROR  not the agent's fault     │
    └─────────────────────────────────────────────────────────────┘

TIMEOUT is the one that bites if you collapse these into a single error string:
a killed process leaves no traceback, so a debug prompt built from `error_text`
would be handed an empty string and regenerate garbage. It needs its own
message ("killed at Ns, make it faster"), which is only possible if the status
distinguishes it.

INFRA_ERROR is likewise not a code defect — it must not be fed to the debug
action, and the search loop counts it toward a separate circuit breaker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol


class RunStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    INFRA_ERROR = "infra_error"

    @property
    def is_agent_fault(self) -> bool:
        """Whether the debug action should try to fix this.

        INFRA_ERROR is the environment's problem; asking the model to repair
        code that never ran wastes a node and teaches the tree nothing.
        """
        return self in (RunStatus.ERROR, RunStatus.TIMEOUT)


@dataclass(frozen=True, slots=True)
class RunResult:
    status: RunStatus
    exit_code: int | None
    duration_s: float
    workspace: Path
    stdout_path: Path | None = None
    stdout_tail: str = ""
    error_text: str = ""
    run_flags: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status is RunStatus.OK

    @property
    def had_network(self) -> bool:
        """Evidence for success criterion 5, read back off the recorded flags."""
        return "--network" not in self.run_flags or "none" not in self.run_flags


class Executor(Protocol):
    def run(self, code: str, *, timeout_s: int, node_id: str) -> RunResult: ...
