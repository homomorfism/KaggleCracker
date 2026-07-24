"""The human approval gate for irreversible actions.

This file names no tool. Whether a call needs a human is read from
``ToolSpec.irreversible`` — never decided by comparing against a hardcoded tool
name. That is deliberate: the gate must stay correct when a fourth tool is added
later, and a test asserts no such name string is present here.

The contrast is the point of the requirement: a reversible action runs straight
through, an irreversible one stops for a human. Do not gate everything.
"""

from core.contracts import err

_APPROVALS = ("yes", "y")


def confirm(preview_text, input_fn=input, output_fn=print):
    """Show what will happen and take exactly one answer. Returns True to proceed.

    Only "yes"/"y" (stripped, case-insensitive) approve. Everything else — a
    blank line, "ok", "sure", "go ahead" — is a denial. Widening this set would
    let a distracted or ambiguous human spend an irreversible action they did
    not mean to, so the accept list stays as narrow as it can be.
    """
    output_fn(preview_text)
    try:
        answer = input_fn("Approve? Type 'yes' to proceed: ")
    except EOFError:
        # No interactive human on the other end (closed/piped stdin). Absence of
        # a person is never consent, so treat it as a denial rather than reading
        # an empty line as approval.
        return False
    return answer.strip().lower() in _APPROVALS


def gated_call(spec, args, input_fn=input, output_fn=print):
    """Run a tool, stopping for human approval only if it is irreversible.

    Returns the tool's own envelope on approval, or a denied_by_human envelope
    on refusal. A denial is a normal error branch, not an exception.
    """
    if not spec.irreversible:
        # Reversible: undoable if wrong, so no human in the loop. Gating it would
        # blur the one distinction requirement 3 is meant to demonstrate.
        return spec.fn(args)

    if not confirm(spec.preview(args), input_fn=input_fn, output_fn=output_fn):
        # retryable=False: the human said no. Re-running the identical call
        # without new information would just ask them the same question again.
        return err("denied_by_human", "human denied the action at the gate", retryable=False)

    return spec.fn(args)
