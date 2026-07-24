"""The agent loop and the single-call dispatcher it runs each tool through.

dispatch() is where the four graded properties meet a real call: validation
before the body, a precheck before the gate, the gate before an irreversible
action, and any raised exception turned into an error branch instead of a
traceback. run_agent() below is hand-written and must stay that way.
"""

from dataclasses import dataclass, field

from core.contracts import err
from core.errors import StepLimitReached
from core.gate import confirm
from core.registry import validate


@dataclass
class RunState:
    # Per-tool count of consecutive failures. A success clears the entry, so a
    # tool that fails, succeeds, then fails again is never wrongly disabled.
    fails: dict = field(default_factory=dict)
    # Tools that have failed too often. run_agent stops offering these by
    # passing the set to registry.schemas(exclude=...).
    disabled: set = field(default_factory=set)


# A tool that fails this many times in a row stops being offered. Two is enough
# to catch a genuinely broken tool without giving up after a single fluke.
_MAX_CONSECUTIVE_FAILS = 2


def dispatch(call, registry, state, input_fn=input, output_fn=print):
    """Run one tool call end to end, always returning an envelope.

    Order is the point of this function; each stage can only be reached if every
    earlier stage passed:
      1. unknown or disabled name -> not_found
      2. validate()               -> bad_input, before the body ever runs
      3. precheck (if any)        -> its own error, still before the gate
      4. gate (if irreversible)   -> denied_by_human on refusal
      5. the tool body            -> wrapped so no exception escapes raw
    """
    name = call.name

    # A disabled tool is treated exactly like one that never existed: it has been
    # removed from the offered schemas, so a call naming it is stale or invented.
    spec = registry.get(name)
    if spec is None or name in state.disabled:
        return _record(state, name, err("not_found", "unknown or disabled tool: %r" % name))

    # Machine-checkable rejection first. A human must never be asked to approve
    # something a string comparison could have thrown out.
    cleaned, verr = validate(spec, call.args)
    if verr is not None:
        return _record(state, name, verr)

    # A deeper, still-mechanical check (e.g. "does this file have the columns we
    # need"). Runs before the gate for the same reason validate does.
    if spec.precheck is not None:
        perr = spec.precheck(cleaned)
        if perr is not None:
            return _record(state, name, perr)

    # The human gate guards irreversible actions only. Reversible ones fall
    # straight through to the call below.
    if spec.irreversible:
        if not confirm(spec.preview(cleaned), input_fn=input_fn, output_fn=output_fn):
            return _record(
                state, name,
                err("denied_by_human", "human denied the action at the gate", retryable=False),
            )

    try:
        result = spec.fn(cleaned)
    except Exception as e:
        # A raised exception is a real failure, not data. We convert it to the
        # exec_failed branch (never swallow it, never let the traceback reach the
        # transcript) so the loop can reason about it like any other error.
        result = err("exec_failed", "%s: %s" % (type(e).__name__, e), retryable=False)

    return _record(state, name, result)


def _record(state, name, result):
    """Update the consecutive-failure bookkeeping and return the result unchanged."""
    if result["ok"]:
        state.fails.pop(name, None)  # a success wipes the streak
    else:
        state.fails[name] = state.fails.get(name, 0) + 1
        if state.fails[name] >= _MAX_CONSECUTIVE_FAILS:
            state.disabled.add(name)
    return result


def tool_message(name, result):
    """Render a tool result for the transcript, ok and error looking different.

    The two shapes are visibly distinct so the model can branch on failure by
    surface form, not by digging into the envelope.
    """
    if result["ok"]:
        return "[TOOL OK %s] %s" % (name, result["data"])
    e = result["error"]
    return "[TOOL ERROR %s] kind=%s retryable=%s msg=%s" % (
        name, e["kind"], e["retryable"], e["msg"],
    )


# --- hand-written; keep it whiteboard-short --------------------------------
# The model is now passed in, not a module global, so a test hands over a
# FakeModel directly and nothing is patched. Three explicit stopping conditions
# live here: an external should_stop, the model choosing to emit no tool calls,
# and the step cap.
def run_agent(messages, model, registry, max_steps=12, should_stop=None,
              input_fn=input, output_fn=print):
    state = RunState()
    for step in range(max_steps):
        if should_stop is not None and should_stop(state, messages):
            return None

        tools = registry.schemas(exclude=tuple(state.disabled))
        reply = model(messages, tools)
        messages.append(reply)

        if not reply.tool_calls:
            return reply.text

        for call in reply.tool_calls:
            result = dispatch(call, registry, state, input_fn, output_fn)
            messages.append(tool_message(call.name, result))

    raise StepLimitReached(messages)
