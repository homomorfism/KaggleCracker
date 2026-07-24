"""Tests for dispatch(), tool_message(), and RunState in core/loop.py.

Each test asserts on the consequence of the dispatch stage it exercises: on a
rejection, that the tool body never ran; on disabling, that the name lands in
state.disabled and a later call is refused as not_found.
"""

from dataclasses import dataclass

import pytest

from core.contracts import is_err
from core.loop import RunState, dispatch, tool_message
from core.registry import Registry, ToolSpec


@dataclass
class Call:
    name: str
    args: dict


def _ok_fn(ran):
    def fn(args):
        ran.append(args)
        return {"ok": True, "data": {"echo": args}}
    return fn


def _spec(name, fn, **kw):
    params = {"x": {"type": "int", "min": 0, "default": 1}}
    return ToolSpec(name, "d", params, fn, **kw)


def _reg(spec):
    r = Registry()
    r.register(spec)
    return r


def _mute(*a):
    return None


# --- ordering: each stage stops the ones after it ---------------------------


def test_unknown_tool_is_not_found():
    reg = Registry()
    state = RunState()
    result = dispatch(Call("nope", {}), reg, state, input_fn=_mute, output_fn=_mute)
    assert is_err(result, "not_found")


def test_disabled_tool_is_not_found_and_body_never_runs():
    ran = []
    reg = _reg(_spec("worker", _ok_fn(ran)))
    state = RunState(disabled={"worker"})
    result = dispatch(Call("worker", {}), reg, state, input_fn=_mute, output_fn=_mute)
    assert is_err(result, "not_found")
    assert ran == []


def test_bad_args_rejected_before_body_runs():
    ran = []
    reg = _reg(_spec("worker", _ok_fn(ran)))
    state = RunState()
    # x has min=0; -1 must be rejected by validate() before fn is called.
    result = dispatch(Call("worker", {"x": -1}), reg, state, input_fn=_mute, output_fn=_mute)
    assert is_err(result, "bad_input")
    assert ran == []


def test_precheck_failure_stops_before_gate_and_body():
    ran = []
    reg = _reg(_spec(
        "worker", _ok_fn(ran),
        precheck=lambda args: {"ok": False, "error": {"kind": "not_found", "msg": "no file", "retryable": False}},
    ))
    state = RunState()
    result = dispatch(Call("worker", {"x": 1}), reg, state, input_fn=_mute, output_fn=_mute)
    assert is_err(result, "not_found")
    assert ran == []


def test_irreversible_denied_returns_denied_and_body_never_runs():
    ran = []
    reg = _reg(_spec(
        "worker", _ok_fn(ran),
        irreversible=True, preview=lambda args: "would act",
    ))
    state = RunState()
    result = dispatch(Call("worker", {"x": 1}), reg, state, input_fn=lambda _p="": "no", output_fn=_mute)
    assert is_err(result, "denied_by_human")
    assert ran == []


def test_irreversible_approved_runs_body():
    ran = []
    reg = _reg(_spec(
        "worker", _ok_fn(ran),
        irreversible=True, preview=lambda args: "would act",
    ))
    state = RunState()
    result = dispatch(Call("worker", {"x": 1}), reg, state, input_fn=lambda _p="": "yes", output_fn=_mute)
    assert result["ok"] is True
    assert ran == [{"x": 1}]


def test_reversible_runs_without_prompting():
    ran = []
    reg = _reg(_spec("worker", _ok_fn(ran)))
    state = RunState()

    def boom(_p=""):
        raise AssertionError("reversible tool must not reach the gate")

    result = dispatch(Call("worker", {"x": 1}), reg, state, input_fn=boom, output_fn=_mute)
    assert result["ok"] is True
    assert ran == [{"x": 1}]


def test_raised_exception_becomes_exec_failed_not_a_traceback():
    def explode(args):
        raise RuntimeError("kaboom")

    reg = _reg(_spec("worker", explode))
    state = RunState()
    result = dispatch(Call("worker", {"x": 1}), reg, state, input_fn=_mute, output_fn=_mute)
    assert is_err(result, "exec_failed")
    assert "kaboom" in result["error"]["msg"]


# --- disabling after two consecutive failures -------------------------------


def test_two_consecutive_failures_disable_the_tool():
    def explode(args):
        raise RuntimeError("boom")

    reg = _reg(_spec("worker", explode))
    state = RunState()

    dispatch(Call("worker", {"x": 1}), reg, state, input_fn=_mute, output_fn=_mute)
    assert "worker" not in state.disabled  # one failure is not enough

    dispatch(Call("worker", {"x": 1}), reg, state, input_fn=_mute, output_fn=_mute)
    assert "worker" in state.disabled  # second consecutive failure disables it

    # And once disabled, the tool is refused as not_found.
    result = dispatch(Call("worker", {"x": 1}), reg, state, input_fn=_mute, output_fn=_mute)
    assert is_err(result, "not_found")


def test_a_success_resets_the_failure_counter():
    calls = {"n": 0}

    def flaky(args):
        calls["n"] += 1
        if calls["n"] == 2:
            return {"ok": True, "data": {}}
        raise RuntimeError("boom")

    reg = _reg(_spec("worker", flaky))
    state = RunState()

    dispatch(Call("worker", {"x": 1}), reg, state, input_fn=_mute, output_fn=_mute)  # fail #1
    assert state.fails.get("worker") == 1
    dispatch(Call("worker", {"x": 1}), reg, state, input_fn=_mute, output_fn=_mute)  # success -> reset
    assert "worker" not in state.fails
    dispatch(Call("worker", {"x": 1}), reg, state, input_fn=_mute, output_fn=_mute)  # fail again, count is 1
    assert state.fails.get("worker") == 1
    assert "worker" not in state.disabled


# --- tool_message formatting ------------------------------------------------


def test_tool_message_success_and_error_look_different():
    ok_line = tool_message("worker", {"ok": True, "data": {"score": 0.9}})
    assert ok_line.startswith("[TOOL OK worker]")

    err_line = tool_message(
        "worker",
        {"ok": False, "error": {"kind": "timeout", "msg": "slow", "retryable": True}},
    )
    assert err_line.startswith("[TOOL ERROR worker]")
    assert "kind=timeout" in err_line
    assert "retryable=True" in err_line
