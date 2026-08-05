"""Full-loop integration on fakes — SUCCESS CRITERION 4.

The design doc originally called for proving executor swappability by reading
the interface and agreeing it looked fine. This file is that claim as a test
instead: the entire search loop runs against `FakeExecutor` and `FakeRuntime`,
with no Docker, no network and no API key, in well under a second.

If the engine ever reaches around its interfaces — imports docker, shells out,
assumes a real provider — these tests stop passing. That is the whole point.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kagglecracker.db.store import LockHeld, ProtocolStore, RunStore, WorkerLock, init_db
from kagglecracker.engine.loop import SearchLoop
from kagglecracker.engine.policy import SearchPolicy
from kagglecracker.engine.ranking import best_node, leaderboard
from kagglecracker.executors.fake import FakeExecutor, FakeOutcome
from kagglecracker.runtime.fake import FakeResponse, FakeRuntime

SUBMISSION = "PassengerId,Transported\n" + "".join(
    f"{i},True\n" for i in ["0013_01", "0018_01", "0019_01", "0021_01", "0023_01"]
)


@pytest.fixture
def env(tmp_path, settings, sample_submission):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    pd.read_csv(sample_submission).to_csv(
        settings.data_dir / "sample_submission.csv", index=False
    )
    conn = init_db(settings.db_path)
    protocol = ProtocolStore(conn).set_current(
        metric_name="accuracy", direction="higher_is_better",
        fold_scheme="StratifiedGroupKFold", n_folds=5, seed=42,
    )
    return settings, conn, protocol


def build_loop(env, *, executor=None, runtime=None, policy=None, events=None):
    settings, conn, protocol = env
    return SearchLoop(
        settings=settings,
        conn=conn,
        protocol=protocol,
        runtime=runtime or FakeRuntime(),
        executor=executor or FakeExecutor(root=settings.runs_dir),
        policy=policy,
        on_event=(events.append if events is not None else (lambda _: None)),
    )


#: Fold spread of a plausible real node. Identical fold scores are now rejected
#: as a leak signature, so a fixture that returns [cv] * 5 is not just unrealistic
#: — it is a run the contract refuses. Offsets sum to zero so the mean stays cv.
_FOLD_OFFSETS = (-0.008, 0.011, 0.002, -0.004, -0.001)


def ok_outcome(cv: float = 0.81) -> FakeOutcome:
    return FakeOutcome.ok(cv, [cv + d for d in _FOLD_OFFSETS], SUBMISSION)


# --------------------------------------------------------------------------
# the loop runs, scores, and stops
# --------------------------------------------------------------------------


def test_loop_runs_to_max_nodes_with_no_docker_and_no_network(env):
    settings, conn, protocol = env
    settings.max_nodes = 5
    loop = build_loop(env, executor=FakeExecutor(root=settings.runs_dir, default=ok_outcome()))
    result = loop.run()

    assert len(result.steps) == 5
    assert result.nodes_scored == 5
    assert "max_nodes" in result.stop_reason
    assert best_node(conn, protocol).cv_score == pytest.approx(0.81)


def test_loop_stops_on_max_usd(env):
    settings, _, _ = env
    settings.max_nodes = 1000
    settings.max_usd = 0.005
    runtime = FakeRuntime(default=FakeResponse(usd=0.001))
    loop = build_loop(
        env,
        runtime=runtime,
        executor=FakeExecutor(root=settings.runs_dir, default=ok_outcome()),
    )
    result = loop.run()
    assert "max_usd" in result.stop_reason
    assert 5 <= len(result.steps) <= 6


def test_loop_stops_on_max_wall_clock(env):
    settings, _, _ = env
    settings.max_nodes = 1000
    settings.max_wall_clock_s = 0
    result = build_loop(env).run()
    assert "max_wall_clock_s" in result.stop_reason
    assert result.steps == []


# --------------------------------------------------------------------------
# circuit breaker — the thing budget caps alone cannot do
# --------------------------------------------------------------------------


def test_breaker_stops_a_run_that_only_produces_infra_errors(env):
    """Without this, 20 identical daemon failures burn the budget and the run
    reports success."""
    settings, _, _ = env
    settings.max_nodes = 100
    settings.max_consecutive_infra_errors = 3
    executor = FakeExecutor(
        root=settings.runs_dir, default=FakeOutcome.infra("Cannot connect to the Docker daemon")
    )
    result = build_loop(env, executor=executor).run()

    assert "circuit breaker" in result.stop_reason
    assert "infra" in result.stop_reason
    assert len(result.steps) == 3


def test_breaker_stops_a_run_that_keeps_crashing(env):
    settings, _, _ = env
    settings.max_nodes = 100
    settings.max_consecutive_failures = 5
    executor = FakeExecutor(root=settings.runs_dir, default=FakeOutcome.crash("ValueError"))
    result = build_loop(env, executor=executor).run()

    assert "circuit breaker" in result.stop_reason
    assert len(result.steps) == 5


def test_one_success_resets_the_failure_streak(env):
    """'5 consecutive failures' must mean genuinely stuck, not five scattered
    failures across an otherwise healthy tree."""
    settings, _, _ = env
    settings.max_nodes = 9
    settings.max_consecutive_failures = 3
    executor = FakeExecutor(
        root=settings.runs_dir,
        scripted=[
            FakeOutcome.crash("a"), FakeOutcome.crash("b"),
            ok_outcome(0.80),
            FakeOutcome.crash("c"), FakeOutcome.crash("d"),
            ok_outcome(0.82),
        ],
        default=ok_outcome(0.83),
    )
    result = build_loop(env, executor=executor).run()
    assert "max_nodes" in result.stop_reason
    assert len(result.steps) == 9


def test_a_generation_failure_creates_no_phantom_node(env):
    """Nothing ran, so there is no code to store — a node with no code would
    pollute the tree summary. It still costs budget: the tokens were spent."""
    settings, conn, _ = env
    settings.max_nodes = 2
    runtime = FakeRuntime(default=FakeResponse.truncated())
    result = build_loop(env, runtime=runtime).run()

    assert all(s.node_id is None for s in result.steps)
    assert conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 0
    assert RunStore(conn).get(result.run_id)["usd_spent"] > 0


# --------------------------------------------------------------------------
# contract enforcement inside the loop
# --------------------------------------------------------------------------


def test_a_bad_submission_becomes_a_debuggable_node(env):
    """The escape hatch the plan asked for: a broken submission is a node the
    agent can fix, not a silent scoring failure."""
    settings, conn, _ = env
    settings.max_nodes = 1
    executor = FakeExecutor(
        root=settings.runs_dir,
        default=FakeOutcome.ok(
            0.81,
            [0.81 + d for d in _FOLD_OFFSETS],
            "PassengerId,WrongName\n0013_01,True\n",
        ),
    )
    result = build_loop(env, executor=executor).run()

    assert result.steps[0].status == "contract_violation"
    node = conn.execute("SELECT * FROM nodes").fetchone()
    assert node["code"]  # the code is kept so debug can repair it
    assert "columns must be exactly" in node["error_text"]


def test_wrong_fold_count_is_a_contract_violation(env):
    settings, _, _ = env
    settings.max_nodes = 1
    executor = FakeExecutor(
        root=settings.runs_dir, default=FakeOutcome.ok(0.81, [0.81, 0.82], SUBMISSION)
    )
    result = build_loop(env, executor=executor).run()
    assert result.steps[0].status == "contract_violation"
    assert "CV protocol" in result.steps[0].detail


def test_run_flags_are_recorded_for_every_node(env):
    """Standing evidence for criterion 5 — no container had network access."""
    settings, conn, _ = env
    settings.max_nodes = 3
    build_loop(
        env, executor=FakeExecutor(root=settings.runs_dir, default=ok_outcome())
    ).run()
    for (flags,) in conn.execute("SELECT run_flags_json FROM nodes"):
        assert '"--network"' in flags and '"none"' in flags


# --------------------------------------------------------------------------
# resume
# --------------------------------------------------------------------------


def test_resume_continues_the_same_run_rather_than_starting_over(env):
    settings, conn, _ = env
    settings.max_nodes = 3
    executor = FakeExecutor(root=settings.runs_dir, default=ok_outcome())
    first = build_loop(env, executor=executor).run()
    assert len(first.steps) == 3

    settings.max_nodes = 6
    conn.execute("UPDATE runs SET max_nodes = 6 WHERE id = ?", (first.run_id,))
    second = build_loop(env, executor=executor).run(resume_run_id=first.run_id)

    assert second.run_id == first.run_id
    assert len(second.steps) == 3  # only the remaining three
    assert RunStore(conn).get(first.run_id)["nodes_done"] == 6
    assert conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 6


def test_resume_picks_up_the_step_counter_not_zero(env):
    settings, conn, _ = env
    settings.max_nodes = 4
    executor = FakeExecutor(root=settings.runs_dir, default=ok_outcome())
    first = build_loop(env, executor=executor).run()

    settings.max_nodes = 6
    conn.execute("UPDATE runs SET max_nodes = 6 WHERE id = ?", (first.run_id,))
    events: list[str] = []
    second = build_loop(env, executor=executor, events=events).run(resume_run_id=first.run_id)

    assert any("resuming run" in e for e in events)
    assert second.steps[0].step == 4


# --------------------------------------------------------------------------
# worker lock
# --------------------------------------------------------------------------


def test_a_live_lock_blocks_a_second_worker(env):
    _, conn, _ = env
    WorkerLock(conn, ttl_s=90).acquire()
    conn.execute("UPDATE worker_lock SET pid = 999999 WHERE id = 1")
    with pytest.raises(LockHeld, match="holds the lock"):
        WorkerLock(conn, ttl_s=90).acquire()


def test_a_stale_lock_is_stealable_so_a_crash_cannot_brick_the_run(env):
    _, conn, _ = env
    lock = WorkerLock(conn, ttl_s=90)
    lock.acquire()
    conn.execute(
        "UPDATE worker_lock SET pid = 999999, heartbeat_at = datetime('now', '-10 minutes') "
        "WHERE id = 1"
    )
    stolen = WorkerLock(conn, ttl_s=90).acquire()
    assert stolen and "stole stale lock" in stolen


def test_the_loop_releases_its_lock_when_it_stops(env):
    settings, conn, _ = env
    settings.max_nodes = 1
    build_loop(env, executor=FakeExecutor(root=settings.runs_dir, default=ok_outcome())).run()
    assert conn.execute("SELECT COUNT(*) FROM worker_lock").fetchone()[0] == 0


def test_stealing_is_reported_loudly(env):
    settings, conn, _ = env
    settings.max_nodes = 1
    WorkerLock(conn, ttl_s=90).acquire()
    conn.execute(
        "UPDATE worker_lock SET pid = 999999, heartbeat_at = datetime('now', '-10 minutes') "
        "WHERE id = 1"
    )
    events: list[str] = []
    build_loop(
        env, executor=FakeExecutor(root=settings.runs_dir, default=ok_outcome()), events=events
    ).run()
    assert any("WARNING" in e and "stale lock" in e for e in events)


# --------------------------------------------------------------------------
# the shape of a real search
# --------------------------------------------------------------------------


def test_a_realistic_search_drafts_improves_debugs_and_ranks(env):
    """One end-to-end pass over the behaviour the tree is supposed to exhibit:
    seeded drafts, a recovery from a crash, and a leaderboard that orders
    correctly with fold spread attached."""
    settings, conn, protocol = env
    settings.max_nodes = 8
    executor = FakeExecutor(
        root=settings.runs_dir,
        scripted=[
            ok_outcome(0.78),                 # draft
            FakeOutcome.crash("KeyError"),    # draft that crashes
            ok_outcome(0.80),                 # draft
            ok_outcome(0.83),                 # improve/debug from here on
        ],
        default=ok_outcome(0.85),
    )
    policy = SearchPolicy(protocol, seed=42, seed_drafts=3, max_depth=5)
    events: list[str] = []
    result = build_loop(env, executor=executor, policy=policy, events=events).run()

    assert [s.decision.action for s in result.steps[:3]] == ["draft"] * 3
    assert any(s.status == "error" for s in result.steps)
    assert result.nodes_scored >= 6

    board = leaderboard(conn, protocol)
    scores = [r.node.cv_score for r in board]
    assert scores == sorted(scores, reverse=True)
    assert all(r.fold_std >= 0 for r in board)
