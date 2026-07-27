"""Tests for the shared submission_outcomes table.

Shared memory is the channel the executor and critic coordinate through, and
the hardest to debug — so every terminal decision must leave a queryable row.
Each test asserts the consequence in the TABLE (the row exists with the right
status and rounds, or does not exist at all), not just the executor's return
value. KC_DB and KC_WORKSPACE point at tmp, so no test touches real state.
"""

import pytest

from agents.executor import run_submission
from core.registry import Registry
from memory import relational
from tools.exec.submit import SUBMIT_TO_KAGGLE


def _mute(*a, **k):
    return None


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("KC_DB", str(tmp_path / "experiments.db"))
    monkeypatch.setenv("KC_WORKSPACE", str(tmp_path))
    relational.init_db()

    data = tmp_path / "data"
    data.mkdir(parents=True)
    (data / "sample_submission.csv").write_text("id,target\n1,0\n2,0\n3,0\n")
    (tmp_path / "submission.csv").write_text("id,target\n1,0.9\n2,0.1\n3,0.5\n")

    reg = Registry()
    reg.register(SUBMIT_TO_KAGGLE)
    return reg, tmp_path


def _proposal(exp_id, submission_rows=3, sample_rows=3):
    return {
        "experiment_id": exp_id,
        "submission_path": "submission.csv",
        "sample_submission_path": "data/sample_submission.csv",
        "message": "S6E7 candidate",
        "submission_rows": submission_rows,
        "sample_rows": sample_rows,
    }


# --- the store on its own ----------------------------------------------------


def test_record_and_query_roundtrip(env):
    relational.record_outcome("exp-1", "denied", 0, created_at="2026-07-27T01:00:00")
    relational.record_outcome("exp-2", "submitted", 1, created_at="2026-07-27T02:00:00")

    rows = relational.query_outcomes()
    assert [r["experiment_id"] for r in rows] == ["exp-1", "exp-2"]  # oldest first

    only = relational.query_outcomes(experiment_id="exp-2")
    assert len(only) == 1
    assert only[0]["status"] == "submitted"
    assert only[0]["rounds"] == 1


def test_off_list_status_is_refused_and_leaves_no_row(env):
    # Same posture as core/contracts.err with an unknown kind: fail loudly at
    # the point of failure, and the consequence is that the table stays clean.
    with pytest.raises(ValueError):
        relational.record_outcome("exp-1", "shrugged", 0)
    assert relational.query_outcomes() == []


# --- the executor writing through it -----------------------------------------


def test_approved_submission_leaves_a_submitted_row(env):
    reg, _ = env
    relational.add_experiment("exp-1", "alice", "lightgbm", 0.88, "done", "2026-07-27")

    run_submission(_proposal("exp-1"), reg,
                   input_fn=lambda _p="": "yes", output_fn=_mute)

    rows = relational.query_outcomes(experiment_id="exp-1")
    assert len(rows) == 1
    assert rows[0]["status"] == "submitted"
    assert rows[0]["rounds"] == 0


def test_denied_submission_leaves_a_denied_row_and_spends_nothing(env):
    reg, ws = env
    relational.add_experiment("exp-1", "alice", "lightgbm", 0.88, "done", "2026-07-27")

    run_submission(_proposal("exp-1"), reg,
                   input_fn=lambda _p="": "no", output_fn=_mute)

    rows = relational.query_outcomes(experiment_id="exp-1")
    assert [r["status"] for r in rows] == ["denied"]
    # The denial is on the record AND nothing was spent — both consequences.
    assert not (ws / "submissions" / "quota.json").exists()


def test_gave_up_row_records_the_rounds_spent(env):
    reg, _ = env
    relational.add_experiment("exp-1", "alice", "lightgbm", 0.88, "done", "2026-07-27")

    run_submission(
        _proposal("exp-1", submission_rows=2, sample_rows=3),
        reg, reviser=lambda proposal, review_obj: proposal,  # never fixes it
        input_fn=lambda _p="": "no", output_fn=_mute,
    )

    rows = relational.query_outcomes(experiment_id="exp-1")
    assert [r["status"] for r in rows] == ["gave_up"]
    assert rows[0]["rounds"] == 2  # exactly the effort budget, then it stopped


def test_escalation_is_on_the_record_too(env):
    reg, _ = env
    # Nothing seeded: the critic escalates because the experiment is missing.
    run_submission(_proposal("ghost"), reg,
                   input_fn=lambda _p="": "no", output_fn=_mute)

    rows = relational.query_outcomes(experiment_id="ghost")
    assert [r["status"] for r in rows] == ["escalated"]
