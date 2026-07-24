"""The tool envelope and its constructors.

Every tool returns exactly one of these shapes and never a bare value:

    {"ok": True,  "data": {...}}
    {"ok": False, "error": {"kind": ..., "msg": ..., "retryable": bool}}

Going through ``ok`` / ``err`` instead of building dicts by hand means the
shape is defined in one place and ``err`` can reject an off-list ``kind`` at
the point of failure, before it ever reaches the loop.
"""

from core.errors import ERROR_KINDS


def ok(**data):
    """Wrap a successful result. Keyword args become the ``data`` payload."""
    return {"ok": True, "data": data}


def err(kind, msg, retryable=False):
    """Wrap a failure.

    ``kind`` must be one of ERROR_KINDS. The assert fires loudly on a typo or an
    improvised kind rather than letting a malformed envelope flow downstream,
    where it would be far harder to trace back to its source.
    """
    assert kind in ERROR_KINDS, "unknown error kind: %r" % (kind,)
    return {"ok": False, "error": {"kind": kind, "msg": msg, "retryable": retryable}}


def is_err(result, kind=None):
    """True if ``result`` is a failure envelope.

    With ``kind`` given, also require that the failure be of that specific kind,
    so a branch can ask "did this fail *because* of X" in one call.
    """
    if result.get("ok", True):
        return False
    if kind is None:
        return True
    return result["error"]["kind"] == kind
