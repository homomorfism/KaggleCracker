"""Closed set of error kinds and the loop's stop signal.

Every ``ok: False`` envelope must carry a ``kind`` drawn from ERROR_KINDS.
The list is closed on purpose: a failure that does not fit one of these is a
signal to stop and think, not to invent a new label the rest of the system has
never seen.
"""

# A tuple, not a list, so it cannot be appended to at runtime by accident.
ERROR_KINDS = (
    "not_found",
    "bad_input",
    "timeout",
    "rate_limited",
    "exec_failed",
    "denied_by_human",
    "quota_exhausted",
)


class StepLimitReached(Exception):
    """Raised when the loop hits its step cap without a stopping condition.

    This is a stop signal, not a tool error: it never becomes an envelope. The
    partial transcript is carried on the exception so the caller can inspect
    what happened up to the cap instead of losing it.
    """

    def __init__(self, messages):
        self.messages = messages
        super().__init__("step limit reached after %d messages" % len(messages))
