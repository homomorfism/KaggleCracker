"""Tests for the SQLite experiment store (memory/relational.py).

KC_DB is pointed at a file under tmp_path via the `db` fixture, so no test ever
writes into the real workspace/ (the same discipline as the KC_WORKSPACE rule in
AGENTS.md). Each test asserts on the *consequence* of a filter — that the rows it
should exclude are genuinely absent — not merely that some rows came back.
"""

import pytest

from memory import relational


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Point KC_DB at a fresh tmp file and create the table. Returns nothing;
    functions read KC_DB, which exercises the env-var default path."""
    monkeypatch.setenv("KC_DB", str(tmp_path / "experiments.db"))
    relational.init_db()


def _seed():
    # Three rows spanning two users, two model types, and a wide cv range, so
    # every filter below has both a row to keep and a row to drop.
    relational.add_experiment("e1", "alice", "xgboost", 0.90, "done", "2026-07-01")
    relational.add_experiment("e2", "alice", "lightgbm", 0.60, "done", "2026-07-02")
    relational.add_experiment("e3", "bob", "xgboost", 0.95, "done", "2026-07-03")


def test_user_id_filters_out_other_users(db):
    _seed()
    rows = relational.query_experiments("alice")
    ids = {r["id"] for r in rows}
    assert ids == {"e1", "e2"}          # both of alice's rows
    assert "e3" not in ids              # bob's row must not leak through


def test_min_cv_drops_low_scores(db):
    _seed()
    rows = relational.query_experiments("alice", min_cv=0.8)
    ids = {r["id"] for r in rows}
    assert ids == {"e1"}                # 0.90 kept
    assert "e2" not in ids             # 0.60 is below the threshold, must drop


def test_model_type_selects_one_model(db):
    _seed()
    rows = relational.query_experiments("alice", model_type="lightgbm")
    ids = {r["id"] for r in rows}
    assert ids == {"e2"}                # the only lightgbm row for alice
    assert "e1" not in ids             # xgboost row must not match


def test_row_is_structured_and_typed(db):
    _seed()
    (row,) = relational.query_experiments("bob")
    assert row == {
        "id": "e3",
        "user_id": "bob",
        "model_type": "xgboost",
        "cv_score": 0.95,
        "status": "done",
        "created_at": "2026-07-03",
    }
    assert isinstance(row["cv_score"], float)  # stored REAL, read back as float


def test_query_binds_values_as_data_not_sql(db):
    """A user_id containing SQL syntax is matched literally, never executed:
    proof the query is parameterised, not string-interpolated."""
    _seed()
    relational.add_experiment(
        "e4", "alice' OR '1'='1", "xgboost", 0.99, "done", "2026-07-04"
    )
    # If the value were interpolated, this WHERE would match every row.
    rows = relational.query_experiments("alice' OR '1'='1")
    assert {r["id"] for r in rows} == {"e4"}
