"""The recon stopping condition: recon is done when the plan is on disk.

Unit tests pin the contract (None while no plan exists, a truthful message
once it does), and the integration test proves the property that matters:
after the plan lands, run_agent stops WITHOUT consulting the model again —
the FakeModel records every time it is called, so an extra step would show.

The workspace is redirected to tmp_path via the `stop_ws` fixture, so no test
writes into the real workspace/ (the KC_WORKSPACE rule in AGENTS.md).
"""

import pytest

from core.errors import StepLimitReached
from core.loop import run_agent
from core.registry import Registry
from tests.fakemodel import FakeModel, Reply, ToolCall
from tools.recon.paths import plan_path
from tools.recon.plans import register as register_plans
from tools.recon.profile import register as register_profile
from tools.recon.stopping import plan_written


def _mute(*a, **k):
    return None


@pytest.fixture
def stop_ws(tmp_path, monkeypatch):
    """Point KC_WORKSPACE at a fresh tmp dir. Returns the workspace root."""
    monkeypatch.setenv("KC_WORKSPACE", str(tmp_path))
    return tmp_path


# --- unit: the condition itself ------------------------------------------------


def test_no_plan_means_no_stop(stop_ws):
    assert plan_written("train.csv")(None, []) is None


def test_existing_plan_stops_with_a_truthful_message(stop_ws):
    path = plan_path("train.csv")
    path.parent.mkdir(parents=True)
    path.write_text("# plan\n")
    message = plan_written("train.csv")(None, [])
    assert message  # truthy -> run_agent stops
    assert str(path) in message


def test_the_condition_is_per_dataset(stop_ws):
    # A plan for another dataset must not stop this run.
    path = plan_path("other.csv")
    path.parent.mkdir(parents=True)
    path.write_text("# plan\n")
    assert plan_written("train.csv")(None, []) is None


# --- integration: the loop actually stops ----------------------------------------


def test_run_agent_stops_after_the_plan_lands_without_asking_the_model_again(stop_ws):
    reg = register_plans(Registry())
    model = FakeModel([
        Reply(tool_calls=[ToolCall("write_preprocessing_plan",
                                   {"dataset": "train.csv",
                                    "plan_markdown": "# plan\n- drop id\n"})]),
        # Never consumed: the stop fires at the top of the next step. If the
        # loop consulted the model again, seen would grow and the assert fails.
        Reply(text="this reply must never be requested"),
    ])
    result = run_agent([], model, reg, max_steps=5,
                       should_stop=plan_written("train.csv"),
                       input_fn=_mute, output_fn=_mute)
    assert result is None                      # stopped by condition, not by text
    assert plan_path("train.csv").exists()     # the plan actually landed
    assert len(model.seen) == 1                # exactly one model consultation


def test_without_the_condition_the_step_cap_is_the_backstop(stop_ws):
    # A model that profiles a missing file forever: two consecutive failures
    # disable the tool, every later call is not_found, and the run must end in
    # StepLimitReached — the explicit cap, never a silent runaway.
    reg = register_profile(Registry())
    call = ToolCall("profile_dataset", {"path": "nope.csv", "checks": ["dtypes"]})
    model = FakeModel([Reply(tool_calls=[call]) for _ in range(4)])
    with pytest.raises(StepLimitReached):
        run_agent([], model, reg, max_steps=4, input_fn=_mute, output_fn=_mute)
