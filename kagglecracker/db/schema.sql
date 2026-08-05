-- KaggleCracker tree store. SQLite, WAL mode.
--
--   protocols ──< nodes >── runs
--        │          │
--        │          └──< submissions   (UNIQUE node_id: one submission per node)
--        │
--        └── is_current = 1 on exactly one row; ranking filters on it, which is
--            how "pre-change node scores are stale" becomes a WHERE clause
--            instead of a manual chore.
--
--   worker_lock is a single row. One worker writes; the web process only reads.
--   A stale lock (heartbeat older than the TTL) is stealable, so a crashed
--   worker cannot brick the run.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- The scoring contract. Changing the fold scheme inserts a NEW row and flips
-- is_current; nothing is mutated or deleted, so the tree keeps a full audit
-- trail of what was scored how.
CREATE TABLE IF NOT EXISTS protocols (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name       TEXT    NOT NULL,
    direction         TEXT    NOT NULL CHECK (direction IN ('higher_is_better', 'lower_is_better')),
    fold_scheme       TEXT    NOT NULL,
    fold_params_json  TEXT    NOT NULL DEFAULT '{}',
    n_folds           INTEGER NOT NULL,
    seed              INTEGER NOT NULL,
    rationale         TEXT    NOT NULL DEFAULT '',
    created_at        TEXT    NOT NULL DEFAULT (datetime('now')),
    is_current        INTEGER NOT NULL DEFAULT 0 CHECK (is_current IN (0, 1))
);

-- Only one protocol may be current at a time.
CREATE UNIQUE INDEX IF NOT EXISTS ix_protocols_current
    ON protocols (is_current) WHERE is_current = 1;

-- One row per worker invocation. Budget accounting and the circuit breaker
-- live here; `nodes` outlives any single run.
CREATE TABLE IF NOT EXISTS runs (
    id                            INTEGER PRIMARY KEY AUTOINCREMENT,
    protocol_id                   INTEGER NOT NULL REFERENCES protocols(id),
    max_nodes                     INTEGER NOT NULL,
    max_wall_clock_s              INTEGER NOT NULL,
    max_usd                       REAL    NOT NULL,
    max_consecutive_failures      INTEGER NOT NULL,
    max_consecutive_infra_errors  INTEGER NOT NULL,
    nodes_done                    INTEGER NOT NULL DEFAULT 0,
    wall_clock_s                  REAL    NOT NULL DEFAULT 0,
    usd_spent                     REAL    NOT NULL DEFAULT 0,
    consecutive_failures          INTEGER NOT NULL DEFAULT 0,
    consecutive_infra_errors      INTEGER NOT NULL DEFAULT 0,
    state                         TEXT    NOT NULL DEFAULT 'running'
                                    CHECK (state IN ('running', 'stopped', 'crashed')),
    stop_reason                   TEXT,
    started_at                    TEXT    NOT NULL DEFAULT (datetime('now')),
    ended_at                      TEXT
);

-- One row per solution attempt. `code` is the full train.py; stdout lives on
-- disk (stdout_path) because it can be megabytes.
CREATE TABLE IF NOT EXISTS nodes (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            INTEGER NOT NULL REFERENCES runs(id),
    protocol_id       INTEGER NOT NULL REFERENCES protocols(id),
    parent_id         INTEGER REFERENCES nodes(id),
    action            TEXT    NOT NULL CHECK (action IN ('draft', 'improve', 'debug')),
    depth             INTEGER NOT NULL DEFAULT 0,
    code              TEXT    NOT NULL,
    -- ok            : ran, parsed, submission validated
    -- error         : non-zero exit, traceback captured
    -- timeout       : SIGKILL at the wall clock, NO traceback exists
    -- truncated     : model hit its token ceiling mid-file (config bug, not agent error)
    -- contract_violation : ran but broke the output contract, or returned non-Python
    -- infra_error   : docker/daemon/image failure — not the agent's fault
    status            TEXT    NOT NULL CHECK (status IN (
                          'ok', 'error', 'timeout', 'truncated',
                          'contract_violation', 'infra_error')),
    error_text        TEXT,
    cv_score          REAL,
    fold_scores_json  TEXT,
    metrics_json      TEXT,
    stdout_path       TEXT,
    run_flags_json    TEXT,          -- evidence for "no container had network access"
    -- The exact prompt this node was generated from. Reconstructing it later is
    -- only approximately faithful, because tree_summary changes as the tree
    -- grows — so a question like "did the model actually see that warning?"
    -- cannot be answered after the fact without storing it.
    prompt            TEXT,
    prompt_chars      INTEGER,
    tokens_in         INTEGER,
    tokens_out        INTEGER,
    usd               REAL,
    created_at        TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS ix_nodes_rank
    ON nodes (protocol_id, status, cv_score);
CREATE INDEX IF NOT EXISTS ix_nodes_parent ON nodes (parent_id);
CREATE INDEX IF NOT EXISTS ix_nodes_run    ON nodes (run_id);

-- UNIQUE(node_id) is the idempotency guard: a double-clicked Approve cannot
-- burn two of the day's submissions on the same node.
CREATE TABLE IF NOT EXISTS submissions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id       INTEGER NOT NULL UNIQUE REFERENCES nodes(id),
    kaggle_ref    TEXT,
    message       TEXT    NOT NULL DEFAULT '',
    public_score  REAL,
    state         TEXT    NOT NULL DEFAULT 'pending'
                    CHECK (state IN ('pending', 'complete', 'rejected', 'error')),
    error_text    TEXT,
    submitted_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- Single row (id is pinned to 1). Starting a second worker fails loudly rather
-- than silently doubling CPU load and forking the tree.
CREATE TABLE IF NOT EXISTS worker_lock (
    id            INTEGER PRIMARY KEY CHECK (id = 1),
    pid           INTEGER NOT NULL,
    hostname      TEXT    NOT NULL,
    run_id        INTEGER REFERENCES runs(id),
    heartbeat_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
