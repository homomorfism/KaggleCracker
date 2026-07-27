"""SQLite domain memory for the recon slice: profile findings and dataset notes.

Two tables, one judgment call about what belongs in a relational store: rows
you will FILTER. A finding ("income: 2 missing of 10") is re-askable with a
fixed query shape — by dataset, by check, by flagged — so it lives here, not in
a transcript and not in a free-form document store. A dataset note is to a
dataset what a review is to a film: user-authored content attached to a domain
entity, always fetched by (dataset, visibility) — again a fixed query shape.
The agent's own free-form musings have no fixed shape and belong to the team's
document store, not to this module.

Visibility is enforced in the SQL WHERE clause: another user's private note is
never SELECTed, not fetched-then-filtered, so it cannot leak past a later bug
in the presentation layer. Every query is parameterised — no caller value is
ever formatted into SQL text.

This module deals in plain rows (dicts). Wrapping them in the {ok, data} /
{ok, error} envelope is the tool layer's job, not the store's.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone

from tools.recon.paths import db_path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS findings (
    dataset       TEXT NOT NULL,
    check_name    TEXT NOT NULL,
    column_name   TEXT NOT NULL,
    value         TEXT NOT NULL,
    flagged       INTEGER NOT NULL,
    rows_profiled INTEGER NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dataset_notes (
    id          TEXT PRIMARY KEY,
    dataset     TEXT NOT NULL,
    column_name TEXT,
    user_id     TEXT NOT NULL,
    shared      INTEGER NOT NULL,
    note        TEXT NOT NULL,
    cue         TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""


def _connect():
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    # Idempotent, so every entry point can call _connect() and the store needs
    # no separate init step a caller could forget.
    conn.executescript(_SCHEMA)
    return conn


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- findings ----------------------------------------------------------------


def _flag(check, entry):
    """Whether one per-column finding needs a decision in the plan (flagged),
    or merely needs to be known (not). Description-only checks return False."""
    if check == "missingness":
        return entry["missing"] > 0
    if check == "dtypes":
        return entry["mixed"]
    if check == "cardinality":
        return entry["likely_id"] or entry["constant"]
    if check == "train_test_drift":
        return entry.get("drifted", False)
    return False  # numeric_summary, correlation_with_target: descriptive


def _rows_for_check(check, payload):
    """Flatten one check's report into (column_name, value, flagged) rows.

    Most checks are per-column dicts and map one key to one row. target_balance
    is keyed by CLASS, not column, so it stays whole under '*'; drift carries
    file-level column-set differences that also land under '*' — flagged when
    the two files disagree about which columns exist at all.
    """
    if check == "target_balance":
        return [("*", payload, False)]
    if check == "train_test_drift":
        head = {
            "only_in_this_file": payload["only_in_this_file"],
            "only_in_compare": payload["only_in_compare"],
        }
        rows = [("*", head, bool(head["only_in_this_file"] or head["only_in_compare"]))]
        rows += [
            (name, entry, _flag(check, entry))
            for name, entry in payload["columns"].items()
        ]
        return rows
    return [(name, entry, _flag(check, entry)) for name, entry in payload.items()]


def record_report(dataset, report, checks):
    """Persist a profile report as findings rows; returns how many were written.

    Replace-per-check, in one transaction: the table always holds the LATEST
    profile of each (dataset, check), so a re-profile can never leave stale
    rows from the previous run mixed in with fresh ones.
    """
    now = _now()
    written = 0
    conn = _connect()
    try:
        with conn:
            for check in checks:
                if check not in report:
                    continue
                conn.execute(
                    "DELETE FROM findings WHERE dataset = ? AND check_name = ?",
                    (dataset, check),
                )
                for column_name, value, flagged in _rows_for_check(check, report[check]):
                    conn.execute(
                        "INSERT INTO findings VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            dataset,
                            check,
                            column_name,
                            json.dumps(value),
                            int(bool(flagged)),
                            report.get("rows_profiled", 0),
                            now,
                        ),
                    )
                    written += 1
    finally:
        conn.close()
    return written


def query_findings(dataset, check_name=None, flagged_only=False):
    """The latest findings for a dataset, as a list of dicts, stable order."""
    sql = (
        "SELECT dataset, check_name, column_name, value, flagged, rows_profiled, "
        "created_at FROM findings WHERE dataset = ?"
    )
    params = [dataset]
    if check_name is not None:
        sql += " AND check_name = ?"
        params.append(check_name)
    if flagged_only:
        sql += " AND flagged = 1"
    sql += " ORDER BY check_name, column_name"

    conn = _connect()
    try:
        out = []
        for row in conn.execute(sql, params):
            d = dict(row)
            d["value"] = json.loads(d["value"])
            d["flagged"] = bool(d["flagged"])
            out.append(d)
        return out
    finally:
        conn.close()


# --- dataset notes -----------------------------------------------------------


def add_note(user_id, dataset, note, cue, column_name=None, shared=False):
    """Append one note and return it as stored (with its id and timestamp)."""
    record = {
        "id": uuid.uuid4().hex,
        "dataset": dataset,
        "column_name": column_name,
        "user_id": user_id,
        "shared": bool(shared),
        "note": note,
        "cue": cue,
        "created_at": _now(),
    }
    conn = _connect()
    try:
        with conn:
            conn.execute(
                "INSERT INTO dataset_notes VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record["id"],
                    record["dataset"],
                    record["column_name"],
                    record["user_id"],
                    int(record["shared"]),
                    record["note"],
                    record["cue"],
                    record["created_at"],
                ),
            )
    finally:
        conn.close()
    return record


def get_notes(user_id, dataset, query=None):
    """Notes on a dataset visible to user_id: their own plus anyone's shared.

    The visibility test is INSIDE the WHERE clause: a private note belonging to
    a different user is never selected, so no later code path can leak it —
    there is nothing to leak. An optional query narrows by case-insensitive
    substring over note text, cue and column name (the cue is the designed
    match target; the note text is included so a cue author's word choice is
    not a single point of failure).
    """
    sql = (
        "SELECT id, dataset, column_name, user_id, shared, note, cue, created_at "
        "FROM dataset_notes WHERE dataset = ? AND (user_id = ? OR shared = 1)"
    )
    params = [dataset, user_id]
    if query is not None:
        sql += (
            " AND instr(lower(note || ' ' || cue || ' ' || coalesce(column_name, '')),"
            " lower(?)) > 0"
        )
        params.append(query)
    sql += " ORDER BY created_at, rowid"

    conn = _connect()
    try:
        out = []
        for row in conn.execute(sql, params):
            d = dict(row)
            d["shared"] = bool(d["shared"])
            out.append(d)
        return out
    finally:
        conn.close()
