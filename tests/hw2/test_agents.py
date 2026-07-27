"""Tests for the critic + executor two-agent coordination.

The agents talk only through the {status, result, needs_approval} object and the
shared experiments table. KC_DB and KC_WORKSPACE are redirected to tmp, so no
test touches real state. Each test asserts on a consequence: the gate was (or
was not) reached, no quota was spent, the loop actually stopped at its cap.
"""

import json

import pytest

from agents.critic import CV_FLOOR, review_submission
from agents.executor import run_submission
from core.registry import Registry
from memory import relational
from tools.exec.submit import SUBMIT_TO_KAGGLE


def _mute(*a, **k):
    return None


def _never_gate(_p=""):
    raise AssertionError("the human gate must not be reached")


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Redirect the shared table and the workspace to tmp, seed a matching
    submission file pair, and return (registry, workspace)."""
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


def _seed_experiment(exp_id, model_type="lightgbm", cv_score=0.88):
    relational.add_experiment(exp_id, "alice", model_type, cv_score, "done", "2026-07-27")


def _proposal(exp_id, submission_rows=3, sample_rows=3):
    return {
        "experiment_id": exp_id,
        "submission_path": "submission.csv",
        "sample_submission_path": "data/sample_submission.csv",
        "message": "S6E7 candidate",
        "submission_rows": submission_rows,
        "sample_rows": sample_rows,
    }


# --- the critic in isolation -------------------------------------------------


def test_bad_row_count_gets_status_revise(env):
    _seed_experiment("exp-1")  # good cv + model_type, so only the rows are wrong
    obj = review_submission(_proposal("exp-1", submission_rows=2, sample_rows=3))
    assert obj["status"] == "revise"
    assert obj["needs_approval"] is False  # a revise stays in the automated loop


def test_clean_proposal_passes_and_still_needs_approval(env):
    _seed_experiment("exp-1")
    obj = review_submission(_proposal("exp-1"))
    assert obj["status"] == "pass"
    assert obj["needs_approval"] is True  # irreversible submit still needs the gate


def test_low_cv_escalates(env):
    _seed_experiment("exp-1", cv_score=CV_FLOOR - 0.01)
    obj = review_submission(_proposal("exp-1"))
    assert obj["status"] == "escalate"
    assert obj["needs_approval"] is True


def test_missing_model_type_escalates(env):
    _seed_experiment("exp-1", model_type="")
    assert review_submission(_proposal("exp-1"))["status"] == "escalate"


def test_missing_experiment_escalates(env):
    # Nothing seeded: the record the agents coordinate through is absent.
    assert review_submission(_proposal("nope"))["status"] == "escalate"


# --- the executor wrapping the gate -----------------------------------------


def test_passing_submission_reaches_the_gate(env):
    reg, ws = env
    _seed_experiment("exp-1")

    gate_prompts = []

    def approve(prompt=""):
        gate_prompts.append(prompt)  # records that the human gate was consulted
        return "yes"

    result = run_submission(_proposal("exp-1"), reg, input_fn=approve, output_fn=_mute)

    assert result["status"] == "submitted"
    assert gate_prompts, "the human gate must have been reached"
    # It went through as a dry run: validated and confirmed, nothing uploaded.
    assert result["submit"]["ok"] is True
    assert result["submit"]["data"]["dry_run"] is True
    assert result["submit"]["data"]["submitted"] is False
    # No live slot spent.
    assert not (ws / "submissions" / "quota.json").exists()


def test_denial_at_the_gate_blocks_the_submission(env):
    reg, ws = env
    _seed_experiment("exp-1")

    result = run_submission(_proposal("exp-1"), reg,
                            input_fn=lambda _p="": "no", output_fn=_mute)

    assert result["status"] == "denied"
    assert result["submit"]["error"]["kind"] == "denied_by_human"
    assert not (ws / "submissions" / "quota.json").exists()  # nothing spent


def test_round_cap_stops_an_infinite_revise_loop(env):
    reg, _ = env
    _seed_experiment("exp-1")  # facts are fine; only the row count is wrong

    calls = {"n": 0}

    def never_fix(proposal, review_obj):
        # A reviser that keeps handing back the same broken proposal. Without the
        # cap this would loop forever.
        calls["n"] += 1
        return proposal

    result = run_submission(
        _proposal("exp-1", submission_rows=2, sample_rows=3),
        reg, reviser=never_fix, input_fn=_never_gate, output_fn=_mute,
    )

    assert result["status"] == "gave_up"
    assert result["rounds"] == 2            # exactly the effort budget
    assert calls["n"] == 2                  # reviser tried twice, then stopped
    assert result["submit"] is None         # never reached submit / the gate
    # Three reviews in total: initial + one after each of the two revises.
    assert len(result["reviews"]) == 3
    assert all(r["status"] == "revise" for r in result["reviews"])
