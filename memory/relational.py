"""SQLite store for structured experiment records (HW2 memory layer).

An experiment has a fixed shape — a score, a model type, a status, a timestamp —
so it belongs in SQLite, not the free-form JSON document store. Free-form facts
go there; rows with columns you will filter and aggregate go here.

Every query is parameterised. No caller value is ever formatted into SQL text,
so a ``user_id`` or ``model_type`` carrying a quote or a ``;`` is bound as data
and can never become part of the query. This module deals in plain rows (dicts);
wrapping them in the {ok, data}/{ok, error} envelope is the tool layer's job, not
the store's.
"""

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

# The one place the column order lives. SELECT and INSERT both read it so they
# can never drift apart.
_COLUMNS = ("id", "user_id", "model_type", "cv_score", "status", "created_at")

# Same discipline for the outcomes table.
_OUTCOME_COLUMNS = ("id", "experiment_id", "status", "rounds", "created_at")

# The closed set of terminal outcomes the executor can record — the same idea
# as ERROR_KINDS in core/errors.py: an off-list status is a wiring bug, refused
# loudly at the point of failure instead of becoming a row nobody can interpret.
OUTCOME_STATUSES = ("submitted", "denied", "escalated", "gave_up")


def _db_path():
    # Tests point KC_DB at a file under tmp_path (same idea as the KC_WORKSPACE
    # convention in the exec slice), so no test ever writes into the real
    # workspace/. Real runs fall back to workspace/experiments.db.
    return os.environ.get("KC_DB", str(Path("workspace") / "experiments.db"))


def init_db(path=None):
    """Create the experiments table if it is not already there.

    Idempotent — safe to call at the start of every run. Returns the path used
    so a caller (or a test) can confirm which file it touched.
    """
    path = path or _db_path()
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id         TEXT PRIMARY KEY,
                user_id    TEXT,
                model_type TEXT,
                cv_score   REAL,
                status     TEXT,
                created_at TEXT
            )
            """
        )
        # Submission outcomes are structured rows too — a status from a closed
        # set, a round count, a timestamp — so they belong here beside the
        # experiments they refer to, not in the free-form document store.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS submission_outcomes (
                id            TEXT PRIMARY KEY,
                experiment_id TEXT,
                status        TEXT,
                rounds        INTEGER,
                created_at    TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()
    return path


def add_experiment(
    id, user_id, model_type, cv_score, status, created_at=None, path=None
):
    """Insert one experiment row. Returns its id.

    created_at defaults to now (UTC, ISO-8601) when the caller does not supply
    one, so ordering by it is always meaningful.
    """
    path = path or _db_path()
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(path)
    try:
        # Six placeholders, six bound values — the row never touches SQL text.
        conn.execute(
            "INSERT INTO experiments (id, user_id, model_type, cv_score, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (id, user_id, model_type, cv_score, status, created_at),
        )
        conn.commit()
    finally:
        conn.close()
    return id


def query_experiments(user_id, min_cv=None, model_type=None, path=None):
    """Return experiment rows for a user, oldest first, as a list of dicts.

    user_id always filters. min_cv (cv_score >= min_cv) and model_type are
    optional and only narrow the result when given. Every condition binds a
    placeholder; the only strings joined into the SQL are the fixed column
    names below, never a caller-supplied value.
    """
    path = path or _db_path()

    clauses = ["user_id = ?"]
    params = [user_id]
    if min_cv is not None:
        clauses.append("cv_score >= ?")
        params.append(min_cv)
    if model_type is not None:
        clauses.append("model_type = ?")
        params.append(model_type)

    sql = (
        "SELECT " + ", ".join(_COLUMNS) + " FROM experiments "
        "WHERE " + " AND ".join(clauses) + " ORDER BY created_at"
    )

    conn = sqlite3.connect(path)
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def record_outcome(experiment_id, status, rounds, created_at=None, path=None):
    """Insert one terminal submission outcome and return its id.

    This is the durable trace of the executor/critic coordination: every
    proposal the executor drives to a terminal state leaves exactly one row
    here, so "what did the agents decide, and after how many rounds?" is a
    query, not an archaeology dig through prose transcripts. status must come
    from OUTCOME_STATUSES — an unknown status is refused at the point of
    failure, mirroring how core/contracts.err refuses an off-list error kind.
    """
    if status not in OUTCOME_STATUSES:
        raise ValueError("unknown outcome status: %r" % (status,))
    path = path or _db_path()
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()
    outcome_id = uuid.uuid4().hex
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "INSERT INTO submission_outcomes (id, experiment_id, status, rounds, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (outcome_id, experiment_id, status, rounds, created_at),
        )
        conn.commit()
    finally:
        conn.close()
    return outcome_id


def query_outcomes(experiment_id=None, path=None):
    """Return outcome rows, oldest first, as a list of dicts.

    With experiment_id given, only that experiment's outcomes. Outcomes are
    coordination records shared by both agents (and read by the offline
    monitor), so unlike experiments there is no per-user filter.
    """
    path = path or _db_path()

    sql = "SELECT " + ", ".join(_OUTCOME_COLUMNS) + " FROM submission_outcomes"
    params = []
    if experiment_id is not None:
        sql += " WHERE experiment_id = ?"
        params.append(experiment_id)
    sql += " ORDER BY created_at"

    conn = sqlite3.connect(path)
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def get_experiment(experiment_id, path=None):
    """Return the single experiment row with this id as a dict, or None.

    This is the shared-memory read the critic and executor agents both use to
    coordinate: experiment facts (cv_score, model_type) come from here, never
    from either agent's prose. The id binds a placeholder like every other value.
    """
    path = path or _db_path()
    conn = sqlite3.connect(path)
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT " + ", ".join(_COLUMNS) + " FROM experiments WHERE id = ?",
            (experiment_id,),
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None
