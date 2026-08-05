"""In-memory executor for tests.

Lets the whole search loop run in under a second with no Docker and no API
cost, which is what turns "SshDockerExecutor is implementable without touching
engine code" from an interface review into a test that either passes or fails.

Scripted outcomes are consumed in order; once exhausted it falls back to
`default`, so a loop test can say "first three nodes crash, then everything
works" without scripting every step.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from kagglecracker.executors.base import RunResult, RunStatus


@dataclass
class FakeOutcome:
    """What the fake should do for one call."""

    status: RunStatus = RunStatus.OK
    metrics: dict | None = None
    submission_csv: str | None = None
    stdout: str = ""
    error_text: str = ""
    duration_s: float = 0.0
    exit_code: int | None = 0

    @classmethod
    def ok(cls, cv_score: float, fold_scores: Iterable[float], submission_csv: str) -> FakeOutcome:
        return cls(
            status=RunStatus.OK,
            metrics={"cv_score": cv_score, "fold_scores": list(fold_scores)},
            submission_csv=submission_csv,
        )

    @classmethod
    def crash(cls, traceback: str) -> FakeOutcome:
        return cls(status=RunStatus.ERROR, error_text=traceback, exit_code=1)

    @classmethod
    def timeout(cls, timeout_s: int = 900) -> FakeOutcome:
        return cls(
            status=RunStatus.TIMEOUT,
            # Mirrors the real executor: a killed process leaves no traceback,
            # so the message has to carry the whole story.
            error_text=f"killed at the {timeout_s}s wall-clock limit, no traceback",
            exit_code=None,
        )

    @classmethod
    def infra(cls, message: str = "Cannot connect to the Docker daemon") -> FakeOutcome:
        return cls(status=RunStatus.INFRA_ERROR, error_text=message, exit_code=None)


# The same flags LocalDockerExecutor emits, so assertions about the security
# posture hold against the fake too.
FAKE_FLAGS = ["--network", "none", "--cap-drop", "ALL", "--security-opt", "no-new-privileges"]


@dataclass
class FakeExecutor:
    root: Path
    scripted: list[FakeOutcome] = field(default_factory=list)
    default: FakeOutcome = field(default_factory=FakeOutcome)
    calls: list[tuple[str, str, int]] = field(default_factory=list)

    def run(self, code: str, *, timeout_s: int, node_id: str) -> RunResult:
        self.calls.append((node_id, code, timeout_s))
        outcome = self.scripted.pop(0) if self.scripted else self.default

        workspace = self.root / node_id
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "train.py").write_text(code)

        # Only a successful run writes output files — mirroring reality, where a
        # crashed or killed script leaves the workspace empty and the contract
        # validators are what turn that into a readable error.
        if outcome.metrics is not None:
            (workspace / "metrics.json").write_text(json.dumps(outcome.metrics))
        if outcome.submission_csv is not None:
            (workspace / "submission.csv").write_text(outcome.submission_csv)

        stdout_path = workspace / "stdout.log"
        stdout_path.write_text(outcome.stdout)

        return RunResult(
            status=outcome.status,
            exit_code=outcome.exit_code,
            duration_s=outcome.duration_s,
            workspace=workspace,
            stdout_path=stdout_path,
            stdout_tail=outcome.stdout,
            error_text=outcome.error_text,
            run_flags=list(FAKE_FLAGS),
        )
