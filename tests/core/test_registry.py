"""Tests for core/registry.py.

Each validate() test asserts on the *consequence* of the check: on rejection,
that no cleaned args come back and the envelope is a bad_input error; on the
happy path, that the returned args are exactly what the tool body would run on.
"""

import pytest

from core.contracts import is_err
from core.registry import Registry, ToolSpec, validate


def _fn(**kwargs):
    return kwargs


@pytest.fixture
def spec():
    params = {
        "path": {"type": "str", "required": True},
        "n": {"type": "int", "min": 1, "max": 100, "default": 10},
        "mode": {"type": "str", "enum": ["fast", "slow"]},
        "rate": {"type": "float", "min": 0.0, "max": 1.0},
        "flag": {"type": "bool", "default": False},
        "checks": {"type": "list", "item_enum": ["schema", "rows"]},
    }
    return ToolSpec("scan_it", "desc", params, _fn)


# --- validate(): one test per branch ---------------------------------------


def test_required_field_missing(spec):
    cleaned, error = validate(spec, {})
    assert cleaned is None
    assert is_err(error, "bad_input")


def test_unknown_parameter_name(spec):
    cleaned, error = validate(spec, {"path": "x", "bogus": 1})
    assert cleaned is None
    assert is_err(error, "bad_input")


def test_enum_violation(spec):
    cleaned, error = validate(spec, {"path": "x", "mode": "medium"})
    assert cleaned is None
    assert is_err(error, "bad_input")


def test_int_below_min(spec):
    cleaned, error = validate(spec, {"path": "x", "n": 0})
    assert cleaned is None
    assert is_err(error, "bad_input")


def test_int_above_max(spec):
    cleaned, error = validate(spec, {"path": "x", "n": 101})
    assert cleaned is None
    assert is_err(error, "bad_input")


def test_wrong_type(spec):
    cleaned, error = validate(spec, {"path": 1})
    assert cleaned is None
    assert is_err(error, "bad_input")


def test_bool_is_not_accepted_as_int(spec):
    # bool subclasses int in Python; the validator must not let True pass as 1.
    cleaned, error = validate(spec, {"path": "x", "n": True})
    assert cleaned is None
    assert is_err(error, "bad_input")


def test_list_item_enum_violation(spec):
    cleaned, error = validate(spec, {"path": "x", "checks": ["nope"]})
    assert cleaned is None
    assert is_err(error, "bad_input")


def test_defaults_applied(spec):
    cleaned, error = validate(spec, {"path": "x"})
    assert error is None
    assert cleaned == {"path": "x", "n": 10, "flag": False}


def test_happy_path_full_and_float_coercion(spec):
    cleaned, error = validate(
        spec,
        {"path": "x", "n": 5, "mode": "fast", "rate": 1, "flag": True, "checks": ["schema"]},
    )
    assert error is None
    # int input to a float param is normalized to float.
    assert isinstance(cleaned["rate"], float) and cleaned["rate"] == 1.0
    assert cleaned["checks"] == ["schema"]


# --- Registry --------------------------------------------------------------


def test_schemas_exclude(spec):
    reg = Registry()
    reg.register(spec)
    reg.register(
        ToolSpec("submit_it", "d", {}, _fn, irreversible=True, preview=lambda **k: "would submit")
    )
    assert [s["name"] for s in reg.schemas()] == ["scan_it", "submit_it"]
    assert [s["name"] for s in reg.schemas(exclude=("submit_it",))] == ["scan_it"]


def test_register_refuses_irreversible_without_preview():
    reg = Registry()
    with pytest.raises(ValueError):
        reg.register(ToolSpec("submit_it", "d", {}, _fn, irreversible=True))


def test_register_refuses_duplicate_name(spec):
    reg = Registry()
    reg.register(spec)
    with pytest.raises(ValueError):
        reg.register(ToolSpec("scan_it", "d", {}, _fn))
