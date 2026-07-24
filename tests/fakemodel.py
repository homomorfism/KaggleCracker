"""A deterministic stand-in for the LLM. No API key, no network.

FakeModel is called exactly like the real model would be — ``model(messages,
tools)`` — and returns the next reply from a script the test wrote. It records
the ``tools`` it was offered on each call so a test can assert that a disabled
tool stopped being presented.
"""

from dataclasses import dataclass, field


@dataclass
class ToolCall:
    name: str
    args: dict = field(default_factory=dict)


@dataclass
class Reply:
    # An empty tool_calls list is how the model signals it is done and hands back
    # its final text.
    text: str = ""
    tool_calls: list = field(default_factory=list)


class FakeModel:
    def __init__(self, replies):
        self._replies = list(replies)
        self.seen = []  # the tools offered on each call, for assertions

    def __call__(self, messages, tools):
        self.seen.append(tools)
        if not self._replies:
            # Running off the end of the script is a test bug, not a model
            # behaviour. Fail loudly instead of hanging or improvising a reply.
            raise AssertionError("FakeModel ran out of scripted replies")
        return self._replies.pop(0)
