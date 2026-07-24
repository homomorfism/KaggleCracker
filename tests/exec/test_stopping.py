"""Tests for the plateau_detector stopping condition.

Each test seeds a leaderboard.jsonl under a tmp workspace and asserts on the
consequence: a reason string when progress has stalled, None while it hasn't or
while there is too little history. The last test drives run_agent itself with a
model that would otherwise loop to the step cap, and shows the detector halts it
first.
"""

import json

import pytest

from core.errors import StepLimitReached
from core.loop import run_agent
from core.registry import Registry
from tests.fakemodel import FakeModel, Reply, ToolCall
from tools.exec.stopping import plateau_detector


def _mute(*a, **k):
    return None


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """Redirect KC_WORKSPACE at a fresh tmp dir. Returns the workspace root."""
    monkeypatch.setenv("KC_WORKSPACE", str(tmp_path))
    return tmp_path


def _seed_leaderboard(ws, scores):
    """Write one leaderboard row per score, in order — the shape
    record_experiment_result writes."""
    exp = ws / "experiments"
    exp.mkdir(parents=True, exist_ok=True)
    rows = [
        json.dumps({"experiment_id": "e%d" % i, "cv_score": s, "fold_scores": [], "notes": ""})
        for i, s in enumerate(scores)
    ]
    (exp / "leaderboard.jsonl").write_text("".join(r + "\n" for r in rows))


# --- the detector in isolation ----------------------------------------------


def test_fires_on_a_flat_series(ws):
    _seed_leaderboard(ws, [0.80, 0.80, 0.80, 0.80])
    reason = plateau_detector(window=3, min_delta=0.0005)(None, [])
    assert reason is not None
    assert "plateau" in reason


def test_fires_on_a_declining_series(ws):
    # Higher-is-better: scores going down are getting worse, so the best has not
    # improved across the window -> plateau.
    _seed_leaderboard(ws, [0.80, 0.79, 0.78, 0.77])
    assert plateau_detector(window=3)(None, []) is not None


def test_quiet_while_scores_improve(ws):
    _seed_leaderboard(ws, [0.80, 0.81, 0.82, 0.83])
    assert plateau_detector(window=3)(None, []) is None


def test_quiet_with_too_few_experiments(ws):
    det = plateau_detector(window=3)
    # No leaderboard at all yet.
    assert det(None, []) is None
    # Fewer experiments than the window.
    _seed_leaderboard(ws, [0.80, 0.80])
    assert det(None, []) is None
    # Exactly the window: still no prior baseline to compare the window against.
    _seed_leaderboard(ws, [0.80, 0.80, 0.80])
    assert det(None, []) is None


def test_respects_lower_is_better_direction(ws, monkeypatch):
    # Flip the confirmed metric direction and the detector must flip with it.
    monkeypatch.setattr("tools.exec.experiments.HIGHER_IS_BETTER", False)
    det = plateau_detector(window=3)
    # Lower-is-better: falling scores ARE improvement -> stay quiet.
    _seed_leaderboard(ws, [0.30, 0.20, 0.10, 0.05])
    assert det(None, []) is None
    # Rising scores are getting worse -> plateau fires.
    _seed_leaderboard(ws, [0.30, 0.31, 0.32, 0.33])
    assert det(None, []) is not None


# --- the detector wired into run_agent --------------------------------------


def test_run_agent_stops_on_plateau_before_max_steps(ws):
    _seed_leaderboard(ws, [0.80, 0.80, 0.80, 0.80])  # already plateaued

    def looping_replies(n):
        # A model that never voluntarily stops: every reply asks to run another
        # experiment, so on its own the loop runs until the step cap.
        return [Reply(tool_calls=[ToolCall("run_experiment", {})]) for _ in range(n)]

    reg = Registry()  # empty: an unknown tool call yields not_found and the loop goes on

    # Baseline: with no stopping condition this model loops to max_steps.
    with pytest.raises(StepLimitReached):
        run_agent([], FakeModel(looping_replies(5)), reg, max_steps=5,
                  input_fn=_mute, output_fn=_mute)

    # With the plateau detector, it stops on the plateau instead of looping.
    reasons = []
    detector = plateau_detector(window=3)

    def recording_should_stop(state, messages):
        r = detector(state, messages)
        if r is not None:
            reasons.append(r)
        return r

    model = FakeModel(looping_replies(5))
    out = run_agent([], model, reg, max_steps=5, should_stop=recording_should_stop,
                    input_fn=_mute, output_fn=_mute)
    assert out is None            # stopped, not a normal text return
    assert model.seen == []       # halted on step 0, before the model was ever consulted
    assert reasons and "plateau" in reasons[0]
