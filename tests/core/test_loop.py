"""dispatch() must turn a *raising precheck* into an error branch.

The tool body is already wrapped so a raised exception becomes exec_failed
instead of a traceback (see test_dispatch.py). The precheck runs earlier, before
the gate, and must behave the same way: a machine check that blows up is a real
failure the loop reasons about, not something that escapes the loop as a raw
traceback (requirement 4 — an error reaches the loop as its own branch).
"""

from dataclasses import dataclass

from core.contracts import is_err
from core.loop import RunState, dispatch
from core.registry import Registry, ToolSpec


@dataclass
class Call:
    name: str
    args: dict


def _mute(*a, **k):
    return None


def _body_must_not_run(args):
    # The precheck raises before the body is reached; if this runs, the ordering
    # is wrong and the test should fail loudly rather than silently pass.
    raise AssertionError("body must not run when the precheck raises")


def test_raising_precheck_becomes_exec_failed_not_a_traceback():
    def exploding_precheck(args):
        raise RuntimeError("precheck kaboom")

    spec = ToolSpec(
        "worker", "d",
        {"x": {"type": "int", "default": 1}},
        _body_must_not_run,
        precheck=exploding_precheck,
    )
    reg = Registry()
    reg.register(spec)
    state = RunState()

    # If dispatch propagated the RuntimeError instead of catching it, this call
    # would raise and the test would error out here rather than assert below.
    result = dispatch(Call("worker", {"x": 1}), reg, state, input_fn=_mute, output_fn=_mute)

    assert is_err(result, "exec_failed")
    assert "precheck kaboom" in result["error"]["msg"]
    assert result["error"]["retryable"] is False
