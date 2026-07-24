"""Tool specs, the registry, and machine-checkable argument validation.

A tool's ``parameters`` is a plain dict mapping each argument name to a small
spec dict:

    {
        "path":  {"type": "str",   "required": True},
        "n":     {"type": "int",   "min": 1, "max": 100, "default": 10},
        "mode":  {"type": "str",   "enum": ["fast", "slow"]},
        "rate":  {"type": "float", "min": 0.0, "max": 1.0},
        "flag":  {"type": "bool",  "default": False},
        "checks":{"type": "list",  "item_enum": ["schema", "rows", "dtypes"]},
    }

``validate`` is the string-comparison gatekeeper the CLAUDE rules require to run
*before* the human gate: anything a comparison can reject is rejected here, and
it returns an envelope rather than raising so a bad argument becomes an ordinary
error branch in the loop instead of a crash.
"""

from dataclasses import dataclass
from typing import Callable, Optional

from core.contracts import err


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    fn: Callable
    irreversible: bool = False
    # A preview renders what an irreversible action *would* do so the human at
    # the gate sees the real consequence before approving. Reversible tools do
    # not need one.
    preview: Optional[Callable] = None
    # An extra machine check the loop can run after validate() but still before
    # the gate — e.g. "does this CSV have the expected columns". Optional.
    precheck: Optional[Callable] = None


class Registry:
    def __init__(self):
        # Insertion order is preserved so schemas() lists tools in a stable,
        # predictable order for the model.
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> ToolSpec:
        # Fail loud at setup time, not inside the loop: an irreversible tool
        # with no preview would send a human to the gate blind, and a duplicate
        # name would silently shadow an existing tool. Both are wiring bugs, so
        # a plain exception at registration is the right, visible failure.
        if spec.name in self._specs:
            raise ValueError("duplicate tool name: %r" % (spec.name,))
        if spec.irreversible and spec.preview is None:
            raise ValueError(
                "irreversible tool %r must define a preview" % (spec.name,)
            )
        self._specs[spec.name] = spec
        return spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._specs.get(name)

    def schemas(self, exclude: tuple = ()) -> list:
        """Tool schemas to hand to the model, minus any excluded names."""
        return [
            {
                "name": s.name,
                "description": s.description,
                "parameters": s.parameters,
            }
            for s in self._specs.values()
            if s.name not in exclude
        ]


def validate(spec: ToolSpec, args: dict):
    """Check ``args`` against ``spec.parameters``.

    Returns ``(cleaned_args, None)`` on success, or ``(None, error_envelope)``
    on the first problem. Never raises.
    """
    # A model can emit anything as its arguments — null, a list, a bare string.
    # Checked first because every line below assumes a mapping, and iterating a
    # non-dict would raise out of a function documented never to raise, turning a
    # malformed call into a crash instead of an ordinary bad_input branch.
    if not isinstance(args, dict):
        return None, err(
            "bad_input", "arguments must be an object, got %s" % type(args).__name__
        )

    params = spec.parameters

    # Unknown names are rejected outright rather than ignored: silently dropping
    # an unexpected argument hides a caller mistake the model should learn from.
    for key in args:
        if key not in params:
            return None, err("bad_input", "unknown parameter: %r" % (key,))

    cleaned = {}
    for pname, pspec in params.items():
        if pname in args:
            value, e = _check(pname, args[pname], pspec)
            if e is not None:
                return None, e
            cleaned[pname] = value
        elif "default" in pspec:
            cleaned[pname] = pspec["default"]
        elif pspec.get("required"):
            return None, err("bad_input", "missing required parameter: %r" % (pname,))
        # Optional with no default: leave it out entirely.

    return cleaned, None


def _check(name, value, pspec):
    """Validate one value. Returns ``(cleaned_value, None)`` or ``(None, env)``."""
    ptype = pspec.get("type")

    if ptype == "str":
        if not isinstance(value, str):
            return None, err("bad_input", "%s must be a string" % name)

    elif ptype == "bool":
        # Checked before int on purpose: bool is a subclass of int in Python, so
        # this branch has to own True/False before the int branch claims them.
        if not isinstance(value, bool):
            return None, err("bad_input", "%s must be a boolean" % name)

    elif ptype == "int":
        # Reject bool here for the same reason: True would otherwise pass as 1.
        if isinstance(value, bool) or not isinstance(value, int):
            return None, err("bad_input", "%s must be an integer" % name)
        e = _check_bounds(name, value, pspec)
        if e is not None:
            return None, e

    elif ptype == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None, err("bad_input", "%s must be a number" % name)
        value = float(value)  # normalize an int input to a float
        e = _check_bounds(name, value, pspec)
        if e is not None:
            return None, e

    elif ptype == "list":
        if not isinstance(value, list):
            return None, err("bad_input", "%s must be a list" % name)
        item_enum = pspec.get("item_enum")
        if item_enum is not None:
            for item in value:
                if item not in item_enum:
                    return None, err(
                        "bad_input",
                        "%s items must be one of %r" % (name, list(item_enum)),
                    )
        return value, None  # enum/bounds below do not apply to lists

    else:
        return None, err("bad_input", "%s has unknown type %r in schema" % (name, ptype))

    # Scalar enum: applies to whichever scalar type declared it.
    enum = pspec.get("enum")
    if enum is not None and value not in enum:
        return None, err("bad_input", "%s must be one of %r" % (name, list(enum)))

    return value, None


def _check_bounds(name, value, pspec):
    lo = pspec.get("min")
    hi = pspec.get("max")
    if lo is not None and value < lo:
        return err("bad_input", "%s must be >= %s" % (name, lo))
    if hi is not None and value > hi:
        return err("bad_input", "%s must be <= %s" % (name, hi))
    return None
