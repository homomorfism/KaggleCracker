"""Drive the real core loop while journaling every event the UI needs.

Nothing in core/ changes: the journal taps the two seams the loop already
exposes — the model callable (replies out, transcript messages in) and the
gate's input_fn/output_fn. Tool results are parsed back out of the transcript
strings core.loop.tool_message writes; that format is ours and stable, so
parsing it is reading our own record, not scraping.
"""

import ast
import re

from core.errors import StepLimitReached
from core.loop import run_agent

_TOOL_OK_RE = re.compile(r"^\[TOOL OK (?P<name>\S+)\] (?P<data>.*)$", re.DOTALL)
_TOOL_ERR_RE = re.compile(
    r"^\[TOOL ERROR (?P<name>\S+)\] kind=(?P<kind>\S+) "
    r"retryable=(?P<retryable>True|False) msg=(?P<msg>.*)$",
    re.DOTALL,
)


class JournalingModel:
    """Wraps the model callable. Before each inner call it journals the tool
    results appended to the transcript since last time; after it, the reply's
    thinking text and tool calls. The caller flushes once more after the loop
    ends to pick up the final step's results."""

    def __init__(self, model, writer):
        self._model = model
        self._writer = writer
        self._seen = 0

    def __call__(self, messages, tools):
        self.flush(messages)
        reply = self._model(messages, tools)
        # A reply with no tool calls is the run's final text; run_finished
        # carries it, so journaling it as "reason" too would show it twice.
        if reply.text and reply.tool_calls:
            self._writer.write("reason", text=reply.text)
        for call in reply.tool_calls:
            self._writer.write("act", tool=call.name, args=call.args)
        return reply

    def flush(self, messages):
        for message in messages[self._seen :]:
            # Reply objects were journaled when produced above; only transcript
            # strings carry new information here.
            if isinstance(message, str):
                self._journal_transcript_line(message)
        self._seen = len(messages)

    def _journal_transcript_line(self, message):
        m = _TOOL_OK_RE.match(message)
        if m:
            try:
                # tool_message renders data with %s — the repr of plain
                # JSON-able values — and literal_eval inverts exactly that.
                data = ast.literal_eval(m.group("data"))
            except (ValueError, SyntaxError):
                # Un-evaluable payloads stay visible as raw text instead of
                # being dropped; the UI shows them, it just cannot chart them.
                data = {"raw": m.group("data")}
            self._writer.write("tool_ok", tool=m.group("name"), data=data)
            return
        m = _TOOL_ERR_RE.match(message)
        if m:
            self._writer.write(
                "tool_error",
                tool=m.group("name"),
                kind=m.group("kind"),
                retryable=m.group("retryable") == "True",
                msg=m.group("msg"),
            )
        # Any other string is the task prompt, already journaled at run_started.


def _gate_output(writer):
    def output_fn(text):
        writer.write("gate_prompt", text=text)

    return output_fn


def _gate_input(writer, answer_fn, scripted):
    def input_fn(prompt):
        # No answer_fn means nobody is wired to the gate; an empty answer is a
        # denial by the gate's own rules — absence of a human is never consent.
        answer = answer_fn() if answer_fn is not None else ""
        # scripted=True marks an answer that came from code (the demo run);
        # live runs pass scripted=False because a person typed it in the UI.
        writer.write("gate_answer", answer=answer, scripted=scripted)
        return answer

    return input_fn


def run_with_journal(messages, model, registry, writer, max_steps=12,
                     gate_answer_fn=None, gate_scripted=True, should_stop=None,
                     prompt=None):
    """run_agent with every observable event mirrored into the journal.

    The writer stays open — the caller owns its lifetime. Failures are
    journaled AND re-raised: the journal records what happened, the caller
    (and the subprocess exit code) still see the real failure. ``prompt``
    overrides what run_started records — the live driver keeps its task in
    the native API conversation, so ``messages`` starts empty there.
    """
    journaling = JournalingModel(model, writer)
    writer.write(
        "run_started",
        prompt=prompt if prompt is not None else (messages[0] if messages else ""),
        max_steps=max_steps,
    )
    try:
        final = run_agent(
            messages,
            journaling,
            registry,
            max_steps=max_steps,
            should_stop=should_stop,
            input_fn=_gate_input(writer, gate_answer_fn, gate_scripted),
            output_fn=_gate_output(writer),
        )
    except StepLimitReached:
        journaling.flush(messages)
        writer.write("run_failed", reason="step limit reached after %d steps" % max_steps)
        raise
    except Exception as e:
        journaling.flush(messages)
        writer.write("run_failed", reason="%s: %s" % (type(e).__name__, e))
        raise
    journaling.flush(messages)
    writer.write("run_finished", text=final or "")
    return final
