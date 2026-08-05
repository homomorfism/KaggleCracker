"""The search loop: pick an action, generate, run, score, persist, repeat.

    ┌─ acquire worker lock (stealable after TTL) ────────────────────────┐
    │                                                                    │
    │   while not stopped:                                               │
    │       check budgets ──► max_nodes / max_wall_clock / max_usd       │
    │       check breaker ──► consecutive failures / infra errors        │
    │           │                                                        │
    │       SearchPolicy.decide(step) ──► (action, parent)               │
    │       PromptBuilder.render      ──► prompt                         │
    │       AgentRuntime.generate     ──► code | truncated | no_code     │
    │       Executor.run              ──► ok | error | timeout | infra   │
    │       contract validators       ──► cv_score | contract_violation  │
    │       persist node, book budget, heartbeat                         │
    │                                                                    │
    └─ release lock, record stop_reason ─────────────────────────────────┘

Two properties the design leans on:

**Resumable.** All state lives in SQLite, so `--resume <run_id>` continues a run
that died at 3am rather than starting over. The step counter is derived from
`nodes_done`, and the policy reseeds from (seed, step), so a resumed run makes
the same decisions an uninterrupted one would have.

**Bounded on four axes, not one.** Budget caps stop a run that is *working*;
the circuit breaker stops one that is *failing*. Without the breaker, twenty
identical infra errors would burn the whole budget and then report a completed
run — technically within budget, entirely useless.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from kagglecracker.config import Settings
from kagglecracker.db.store import NodeStore, Protocol, RunStore, WorkerLock
from kagglecracker.engine.contract import ContractError, SubmissionValidator, parse_metrics
from kagglecracker.engine.policy import Decision, SearchPolicy
from kagglecracker.engine.prompts import PromptBuilder
from kagglecracker.executors.base import Executor, RunStatus
from kagglecracker.runtime.base import AgentRuntime, GenerateStatus


@dataclass(frozen=True, slots=True)
class StepOutcome:
    step: int
    decision: Decision
    node_id: int | None
    status: str
    cv_score: float | None
    usd: float
    duration_s: float
    detail: str = ""


@dataclass
class LoopResult:
    run_id: int
    steps: list[StepOutcome] = field(default_factory=list)
    stop_reason: str = ""

    @property
    def nodes_scored(self) -> int:
        return sum(1 for s in self.steps if s.cv_score is not None)

    @property
    def usd_spent(self) -> float:
        return sum(s.usd for s in self.steps)


class SearchLoop:
    def __init__(
        self,
        *,
        settings: Settings,
        conn: sqlite3.Connection,
        protocol: Protocol,
        runtime: AgentRuntime,
        executor: Executor,
        policy: SearchPolicy | None = None,
        on_event: Callable[[str], None] = lambda _: None,
    ) -> None:
        self.settings = settings
        self.conn = conn
        self.protocol = protocol
        self.runtime = runtime
        self.executor = executor
        self.policy = policy or SearchPolicy(protocol, seed=protocol.seed)
        self.on_event = on_event

        self.nodes = NodeStore(conn)
        self.runs = RunStore(conn)
        self.lock = WorkerLock(conn, ttl_s=settings.worker_lock_ttl_s)
        self.builder = PromptBuilder(settings, protocol)
        self._validator: SubmissionValidator | None = None

    # -- validators -------------------------------------------------------

    @property
    def validator(self) -> SubmissionValidator:
        if self._validator is None:
            self._validator = SubmissionValidator(
                self.settings.data_dir / "sample_submission.csv"
            )
        return self._validator

    # -- stop conditions --------------------------------------------------

    def _stop_reason(self, run: sqlite3.Row, elapsed_s: float) -> str | None:
        if run["nodes_done"] >= run["max_nodes"]:
            return f"reached max_nodes ({run['max_nodes']})"
        if elapsed_s >= run["max_wall_clock_s"]:
            return f"reached max_wall_clock_s ({run['max_wall_clock_s']}s)"
        if run["usd_spent"] >= run["max_usd"]:
            return f"reached max_usd (${run['max_usd']:.2f}, spent ${run['usd_spent']:.4f})"
        # The breaker: a run that is failing identically is not a run worth
        # finishing, however much budget is left.
        if run["consecutive_infra_errors"] >= run["max_consecutive_infra_errors"]:
            return (
                f"circuit breaker: {run['consecutive_infra_errors']} consecutive infra "
                f"errors — the environment is broken, not the code"
            )
        if run["consecutive_failures"] >= run["max_consecutive_failures"]:
            return (
                f"circuit breaker: {run['consecutive_failures']} consecutive failures "
                f"with no scored node"
            )
        return None

    # -- one step ---------------------------------------------------------

    def step(self, step_index: int) -> StepOutcome:
        started = time.monotonic()
        decision = self.policy.decide(self.conn, step=step_index)
        self.on_event(f"step {step_index}: {decision}")

        history = self.nodes.for_protocol(
            self.protocol.id, limit=self.settings.tree_summary_max_nodes
        )
        prompt = self.builder.render(
            decision.action, nodes=history, parent=decision.parent
        )
        gen = self.runtime.generate(prompt, action=decision.action)

        if not gen.ok:
            # Nothing ran, so there is no node to store — storing one would put
            # a phantom in the tree with no code and pollute the summary. It
            # still counts against the budget: the tokens were spent.
            self.runs.record_node(
                self.run_id,
                usd=gen.usd,
                wall_clock_s=time.monotonic() - started,
                status="infra_error" if gen.status is GenerateStatus.REFUSAL else "error",
            )
            self.on_event(f"  generate {gen.status}: {gen.error_text[:160]}")
            return StepOutcome(
                step=step_index,
                decision=decision,
                node_id=None,
                status=str(gen.status),
                cv_score=None,
                usd=gen.usd,
                duration_s=time.monotonic() - started,
                detail=gen.error_text,
            )

        node_id = f"r{self.run_id}-s{step_index}-{decision.action}"
        result = self.executor.run(
            gen.code, timeout_s=self.settings.node_timeout_s, node_id=node_id
        )

        status = str(result.status)
        error_text = result.error_text
        cv_score: float | None = None
        fold_scores: list[float] = []
        extra = None

        if result.status is RunStatus.OK:
            try:
                metrics = parse_metrics(
                    result.workspace, expected_folds=self.protocol.n_folds
                )
                self.validator.validate(result.workspace)
                cv_score, fold_scores, extra = (
                    metrics.cv_score,
                    metrics.fold_scores,
                    metrics.extra,
                )
                error_text = ""
            except ContractError as exc:
                status, error_text = "contract_violation", str(exc)

        stored = self.nodes.insert(
            run_id=self.run_id,
            protocol_id=self.protocol.id,
            parent_id=decision.parent.id if decision.parent else None,
            action=decision.action,
            depth=(decision.parent.depth + 1) if decision.parent else 0,
            code=gen.code,
            status=status,
            error_text=error_text or None,
            cv_score=cv_score,
            fold_scores=fold_scores or None,
            metrics=extra,
            stdout_path=str(result.stdout_path) if result.stdout_path else None,
            run_flags=result.run_flags,
            prompt=prompt,
        prompt_chars=gen.prompt_chars,
            tokens_in=gen.usage.tokens_in,
            tokens_out=gen.usage.tokens_out,
            usd=gen.usd,
        )

        self.runs.record_node(
            self.run_id,
            usd=gen.usd,
            wall_clock_s=time.monotonic() - started,
            status=status,
        )
        self.lock.heartbeat()

        score_text = f"cv {cv_score:.5f}" if cv_score is not None else error_text[:120]
        self.on_event(f"  node {stored.id}: {status}  {score_text}")

        return StepOutcome(
            step=step_index,
            decision=decision,
            node_id=stored.id,
            status=status,
            cv_score=cv_score,
            usd=gen.usd,
            duration_s=time.monotonic() - started,
            detail=error_text,
        )

    # -- the loop ---------------------------------------------------------

    def run(self, *, resume_run_id: int | None = None) -> LoopResult:
        if resume_run_id is None:
            self.run_id = self.runs.create(self.protocol.id, self.settings)
            start_step = 0
        else:
            self.run_id = resume_run_id
            row = self.runs.get(self.run_id)
            # Deriving the step from nodes_done is what makes resume seamless:
            # the policy reseeds from (seed, step), so it makes the same
            # decisions an uninterrupted run would have made.
            start_step = row["nodes_done"]
            self.on_event(
                f"resuming run {self.run_id} at step {start_step} "
                f"(${row['usd_spent']:.4f} spent)"
            )

        stolen = self.lock.acquire(self.run_id)
        if stolen:
            self.on_event(f"WARNING: {stolen}")

        result = LoopResult(run_id=self.run_id)
        started = time.monotonic()

        try:
            step_index = start_step
            while True:
                run = self.runs.get(self.run_id)
                elapsed = time.monotonic() - started
                if (reason := self._stop_reason(run, elapsed)) is not None:
                    result.stop_reason = reason
                    break
                result.steps.append(self.step(step_index))
                step_index += 1
        except KeyboardInterrupt:
            result.stop_reason = "interrupted"
        finally:
            state = "stopped" if result.stop_reason else "crashed"
            self.runs.finish(
                self.run_id, state=state, stop_reason=result.stop_reason or "crashed"
            )
            self.lock.release()

        self.on_event(f"stopped: {result.stop_reason}")
        return result


def workspace_root(settings: Settings) -> Path:
    return settings.runs_dir
