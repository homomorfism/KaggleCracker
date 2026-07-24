"""Tests for core/gate.py.

Every test asserts on the consequence of the gate decision: a denial produces a
denied_by_human envelope AND does not run the tool body; an approval runs it.
"""

import inspect

import pytest

import core.gate as gate
from core.contracts import is_err
from core.registry import ToolSpec


def _canned(answer):
    """An input_fn that ignores the prompt and returns a fixed answer."""
    return lambda _prompt="": answer


def _irreversible_spec(fn):
    return ToolSpec(
        name="do_the_thing",
        description="d",
        parameters={},
        fn=fn,
        irreversible=True,
        preview=lambda args: "would do the thing",
    )


@pytest.mark.parametrize("answer", ["yes", "y", "YES", " Yes "])
def test_these_answers_approve(answer):
    assert confirm_result(answer) is True


@pytest.mark.parametrize("answer", ["", "ok", "sure", "go ahead", "n", "no", "yep", "1"])
def test_these_answers_deny(answer):
    assert confirm_result(answer) is False


def confirm_result(answer):
    return gate.confirm("preview", input_fn=_canned(answer), output_fn=lambda *a: None)


def test_eof_is_a_denial():
    def raise_eof(_prompt=""):
        raise EOFError

    assert gate.confirm("preview", input_fn=raise_eof, output_fn=lambda *a: None) is False


def test_denial_returns_denied_by_human_envelope():
    ran = []
    spec = _irreversible_spec(lambda args: ran.append(True))

    result = gate.gated_call(spec, {}, input_fn=_canned("no"), output_fn=lambda *a: None)

    assert is_err(result, "denied_by_human")
    assert result["error"]["retryable"] is False
    # Consequence: the tool body never ran on a denial.
    assert ran == []


def test_approval_runs_the_tool():
    ran = []

    def fn(args):
        ran.append(True)
        return {"ok": True, "data": {}}

    spec = _irreversible_spec(fn)
    result = gate.gated_call(spec, {}, input_fn=_canned("yes"), output_fn=lambda *a: None)

    assert result["ok"] is True
    assert ran == [True]


def test_reversible_action_is_ungated():
    def boom(_prompt=""):
        raise AssertionError("a reversible action must not prompt for approval")

    spec = ToolSpec("read_something", "d", {}, fn=lambda args: {"ok": True, "data": {}})
    # input_fn raises if touched; passing without a denial proves the gate was skipped.
    result = gate.gated_call(spec, {}, input_fn=boom, output_fn=lambda *a: None)
    assert result["ok"] is True


def test_no_tool_name_hardcoded_in_gate():
    # CLAUDE.md: the gate reads ToolSpec.irreversible, it never keys off a name.
    source = inspect.getsource(gate)
    assert "submit" not in source
