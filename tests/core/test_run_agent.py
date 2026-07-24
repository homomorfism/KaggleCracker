"""Integration tests for run_agent, driven by FakeModel passed in directly.

No monkeypatching of core.loop.model: the model is a parameter now, so each test
hands over its own scripted FakeModel. Every test asserts on a consequence — the
tool ran, the transcript got the right line, or the loop stopped for the stated
reason.
"""

import pytest

from core.errors import StepLimitReached
from core.loop import run_agent
from core.registry import Registry, ToolSpec
from tests.fakemodel import FakeModel, Reply, ToolCall


def _mute(*a):
    return None


def _reg_with_worker(fn):
    reg = Registry()
    reg.register(ToolSpec("worker", "d", {"x": {"type": "int", "default": 1}}, fn))
    return reg


def test_stops_when_model_emits_no_tool_calls():
    reg = _reg_with_worker(lambda a: {"ok": True, "data": {}})
    model = FakeModel([Reply(text="all done", tool_calls=[])])
    out = run_agent([], model, reg, input_fn=_mute, output_fn=_mute)
    assert out == "all done"


def test_runs_a_tool_then_finishes():
    ran = []
    reg = _reg_with_worker(lambda a: (ran.append(a) or {"ok": True, "data": a}))
    model = FakeModel([
        Reply(tool_calls=[ToolCall("worker", {"x": 2})]),
        Reply(text="finished", tool_calls=[]),
    ])
    messages = []
    out = run_agent(messages, model, reg, input_fn=_mute, output_fn=_mute)
    assert out == "finished"
    assert ran == [{"x": 2}]  # the tool body actually ran with validated args
    # a visibly-OK line reached the transcript
    assert any(isinstance(m, str) and m.startswith("[TOOL OK worker]") for m in messages)


def test_step_limit_is_reached_when_model_never_stops():
    reg = _reg_with_worker(lambda a: {"ok": True, "data": {}})
    model = FakeModel([Reply(tool_calls=[ToolCall("worker", {})]) for _ in range(5)])
    with pytest.raises(StepLimitReached):
        run_agent([], model, reg, max_steps=3, input_fn=_mute, output_fn=_mute)


def test_should_stop_halts_before_the_model_is_called():
    reg = _reg_with_worker(lambda a: {"ok": True, "data": {}})
    model = FakeModel([Reply(tool_calls=[ToolCall("worker", {})])])
    out = run_agent(
        [], model, reg,
        should_stop=lambda state, messages: True,
        input_fn=_mute, output_fn=_mute,
    )
    assert out is None
    assert model.seen == []  # stopped on step 0, model never consulted


def test_repeated_failure_disables_tool_so_it_stops_being_offered():
    def explode(a):
        raise RuntimeError("boom")

    reg = _reg_with_worker(explode)
    model = FakeModel([
        Reply(tool_calls=[ToolCall("worker", {})]),  # fail #1
        Reply(tool_calls=[ToolCall("worker", {})]),  # fail #2 -> worker disabled
        Reply(text="giving up", tool_calls=[]),       # model stops
    ])
    out = run_agent([], model, reg, input_fn=_mute, output_fn=_mute)
    assert out == "giving up"
    # first two calls were offered the worker tool; after two failures it is gone.
    assert [s["name"] for s in model.seen[0]] == ["worker"]
    assert [s["name"] for s in model.seen[1]] == ["worker"]
    assert model.seen[2] == []
