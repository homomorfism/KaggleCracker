"""SQLite access. One writer (the worker), many readers (the web process).

WAL mode is load-bearing, not a tuning knob: without it the web process's reads
block the worker's writes and you get `database is locked` mid-run.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def connect(db_path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    if read_only:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(db_path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if not read_only:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
    return conn


#: Columns added after the first databases were created. `CREATE TABLE IF NOT
#: EXISTS` will not add them to an existing file, so they are applied by hand.
MIGRATIONS: tuple[tuple[str, str, str], ...] = (
    ("nodes", "prompt", "ALTER TABLE nodes ADD COLUMN prompt TEXT"),
)


def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(db_path)
    conn.executescript(SCHEMA_PATH.read_text())
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    for table, column, ddl in MIGRATIONS:
        cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            with conn:
                conn.execute(ddl)


@dataclass(frozen=True, slots=True)
class Protocol:
    """The scoring contract: how a node's cv_score is produced and compared.

    Every node carries the protocol_id it was scored under. Ranking filters on
    the current protocol, so changing the fold scheme retires old scores
    automatically instead of silently ranking leaky nodes at the top.
    """

    id: int
    metric_name: str
    direction: str
    fold_scheme: str
    fold_params: dict
    n_folds: int
    seed: int
    rationale: str

    @property
    def higher_is_better(self) -> bool:
        return self.direction == "higher_is_better"

    def describe(self) -> str:
        """One line, injected into every prompt."""
        params = ", ".join(f"{k}={v}" for k, v in sorted(self.fold_params.items()))
        tail = f" ({params})" if params else ""
        return (
            f"{self.metric_name}, {self.direction}, "
            f"{self.n_folds}-fold {self.fold_scheme}{tail}, seed={self.seed}"
        )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Protocol:
        return cls(
            id=row["id"],
            metric_name=row["metric_name"],
            direction=row["direction"],
            fold_scheme=row["fold_scheme"],
            fold_params=json.loads(row["fold_params_json"]),
            n_folds=row["n_folds"],
            seed=row["seed"],
            rationale=row["rationale"],
        )


@dataclass(frozen=True, slots=True)
class Node:
    id: int
    run_id: int
    protocol_id: int
    parent_id: int | None
    action: str
    depth: int
    code: str
    status: str
    error_text: str | None
    cv_score: float | None
    fold_scores: list[float]
    stdout_path: str | None
    prompt: str | None
    prompt_chars: int | None
    tokens_in: int | None
    tokens_out: int | None
    usd: float | None
    created_at: str

    @property
    def scored(self) -> bool:
        return self.status == "ok" and self.cv_score is not None

    @property
    def failure_signature(self) -> str:
        """Coarse fingerprint used to group repeated failures in the tree summary.

        The first line of the error is enough to tell "OOM again" from "another
        KeyError" without turning the digest into a wall of tracebacks.
        """
        first = (self.error_text or "").strip().splitlines()
        return f"{self.status}: {first[-1][:160]}" if first else self.status

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Node:
        return cls(
            id=row["id"],
            run_id=row["run_id"],
            protocol_id=row["protocol_id"],
            parent_id=row["parent_id"],
            action=row["action"],
            depth=row["depth"],
            code=row["code"],
            status=row["status"],
            error_text=row["error_text"],
            cv_score=row["cv_score"],
            fold_scores=json.loads(row["fold_scores_json"]) if row["fold_scores_json"] else [],
            stdout_path=row["stdout_path"],
            prompt=row["prompt"] if "prompt" in row.keys() else None,
            prompt_chars=row["prompt_chars"],
            tokens_in=row["tokens_in"],
            tokens_out=row["tokens_out"],
            usd=row["usd"],
            created_at=row["created_at"],
        )


class NodeStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def insert(self, **fields) -> Node:
        fold_scores = fields.pop("fold_scores", None)
        metrics = fields.pop("metrics", None)
        run_flags = fields.pop("run_flags", None)
        payload = {
            **fields,
            "fold_scores_json": json.dumps(fold_scores) if fold_scores is not None else None,
            "metrics_json": json.dumps(metrics) if metrics is not None else None,
            "run_flags_json": json.dumps(run_flags) if run_flags is not None else None,
        }
        cols = ", ".join(payload)
        marks = ", ".join("?" for _ in payload)
        with self.conn:
            cur = self.conn.execute(
                f"INSERT INTO nodes ({cols}) VALUES ({marks})", tuple(payload.values())
            )
        return self.get(cur.lastrowid)

    def get(self, node_id: int) -> Node:
        row = self.conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        if row is None:
            raise KeyError(f"no node {node_id}")
        return Node.from_row(row)

    def for_protocol(self, protocol_id: int, *, limit: int | None = None) -> list[Node]:
        """Every node scored under this protocol, across ALL runs.

        Deliberately not scoped to a run. Scoping to `run_id` would give the
        agent amnesia between worker restarts — it would redraft models it
        already tried last night. `runs` is the unit of budget accounting;
        `protocols` is the unit of comparability, and therefore of memory.
        """
        sql = "SELECT * FROM nodes WHERE protocol_id = ? ORDER BY id DESC"
        params: tuple = (protocol_id,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (protocol_id, limit)
        return [Node.from_row(r) for r in self.conn.execute(sql, params)]


class RunStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def create(self, protocol_id: int, settings) -> int:
        with self.conn:
            cur = self.conn.execute(
                """
                INSERT INTO runs (protocol_id, max_nodes, max_wall_clock_s, max_usd,
                                  max_consecutive_failures, max_consecutive_infra_errors)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    protocol_id,
                    settings.max_nodes,
                    settings.max_wall_clock_s,
                    settings.max_usd,
                    settings.max_consecutive_failures,
                    settings.max_consecutive_infra_errors,
                ),
            )
        return cur.lastrowid

    def record_node(
        self,
        run_id: int,
        *,
        usd: float,
        wall_clock_s: float,
        status: str,
    ) -> None:
        """Book one node against the run's budget and update the failure streaks.

        The streaks are the circuit breaker's input. They reset on any success,
        so "5 consecutive failures" means a genuinely stuck run rather than five
        scattered ones across a healthy tree.
        """
        ok = status == "ok"
        infra = status == "infra_error"
        with self.conn:
            self.conn.execute(
                """
                UPDATE runs
                   SET nodes_done               = nodes_done + 1,
                       usd_spent                = usd_spent + ?,
                       wall_clock_s             = wall_clock_s + ?,
                       consecutive_failures     = CASE WHEN ? THEN 0
                                                       ELSE consecutive_failures + 1 END,
                       consecutive_infra_errors = CASE WHEN ? THEN consecutive_infra_errors + 1
                                                       ELSE 0 END
                 WHERE id = ?
                """,
                (usd, wall_clock_s, ok, infra, run_id),
            )

    def finish(self, run_id: int, *, state: str, stop_reason: str) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE runs SET state = ?, stop_reason = ?, ended_at = datetime('now') "
                "WHERE id = ?",
                (state, stop_reason, run_id),
            )

    def get(self, run_id: int) -> sqlite3.Row:
        row = self.conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"no run {run_id}")
        return row


class LockHeld(RuntimeError):
    """Another worker owns the lock and is still alive."""


class WorkerLock:
    """Single-writer guard, with a heartbeat so a dead worker cannot brick the run.

    "One experiment at a time" is structural here: there is exactly one worker
    process, so the guarantee is not a config value anyone can set to 2. This
    lock exists to make a *second* worker fail loudly rather than silently
    doubling CPU load and forking the tree.

    The heartbeat is the other half. A lock with no TTL turns a crashed worker
    into a permanently locked project, fixable only by hand-editing the
    database — so a lock whose heartbeat has gone stale is stealable, and the
    steal is logged loudly because it should be rare enough to notice.
    """

    def __init__(self, conn: sqlite3.Connection, *, ttl_s: int = 90) -> None:
        self.conn = conn
        self.ttl_s = ttl_s

    def _current(self) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT *, (julianday('now') - julianday(heartbeat_at)) * 86400 AS age_s "
            "FROM worker_lock WHERE id = 1"
        ).fetchone()

    def acquire(self, run_id: int | None = None) -> str | None:
        """Take the lock. Returns a message if a stale lock was stolen.

        Raises LockHeld if a live worker owns it.
        """
        import os
        import socket

        row = self._current()
        stolen = None
        if row is not None:
            age = row["age_s"] or 0.0
            if age < self.ttl_s and row["pid"] != os.getpid():
                raise LockHeld(
                    f"worker pid {row['pid']} on {row['hostname']} holds the lock "
                    f"(heartbeat {age:.0f}s ago, TTL {self.ttl_s}s). Only one worker may "
                    f"run at a time — a second would double CPU load and fork the tree."
                )
            if row["pid"] != os.getpid():
                stolen = (
                    f"stole stale lock from pid {row['pid']} on {row['hostname']} "
                    f"(last heartbeat {age:.0f}s ago, past the {self.ttl_s}s TTL)"
                )

        with self.conn:
            self.conn.execute(
                """
                INSERT INTO worker_lock (id, pid, hostname, run_id, heartbeat_at)
                VALUES (1, ?, ?, ?, datetime('now'))
                ON CONFLICT(id) DO UPDATE SET
                    pid = excluded.pid,
                    hostname = excluded.hostname,
                    run_id = excluded.run_id,
                    heartbeat_at = datetime('now')
                """,
                (os.getpid(), socket.gethostname(), run_id),
            )
        return stolen

    def heartbeat(self) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE worker_lock SET heartbeat_at = datetime('now') WHERE id = 1"
            )

    def release(self) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM worker_lock WHERE id = 1")


class ProtocolStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def current(self) -> Protocol | None:
        row = self.conn.execute("SELECT * FROM protocols WHERE is_current = 1").fetchone()
        return Protocol.from_row(row) if row else None

    def set_current(
        self,
        *,
        metric_name: str,
        direction: str,
        fold_scheme: str,
        n_folds: int,
        seed: int,
        fold_params: dict | None = None,
        rationale: str = "",
    ) -> Protocol:
        """Insert a protocol and make it current, retiring the previous one.

        Old nodes are neither mutated nor deleted — they simply stop matching
        the ranking filter, which is what "marked stale" means in practice.
        """
        with self.conn:
            self.conn.execute("UPDATE protocols SET is_current = 0 WHERE is_current = 1")
            cur = self.conn.execute(
                """
                INSERT INTO protocols
                    (metric_name, direction, fold_scheme, fold_params_json,
                     n_folds, seed, rationale, is_current)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    metric_name,
                    direction,
                    fold_scheme,
                    json.dumps(fold_params or {}, sort_keys=True),
                    n_folds,
                    seed,
                    rationale,
                ),
            )
        row = self.conn.execute(
            "SELECT * FROM protocols WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
        return Protocol.from_row(row)
